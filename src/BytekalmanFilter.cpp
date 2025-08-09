#include "BytekalmanFilter.h"
#include <Eigen/Cholesky>

namespace byte_kalman
{
	// 卡方检验临界值表（95%置信度），索引对应自由度
	const double ByteKalmanFilter::chi2inv95[10] = {
	0,
	3.8415,
	5.9915,
	7.8147,
	9.4877,
	11.070,
	12.592,
	14.067,
	15.507,
	16.919
	};
	
	// 构造函数：初始化卡尔曼滤波器参数
	ByteKalmanFilter::ByteKalmanFilter()
	{
		int ndim = 4;         // 状态维度
		double dt = 1.;       // 时间间隔

		// 初始化状态转移矩阵（8x8）
		// 状态向量为 [x, y, a, h, vx, vy, va, vh]
		// 其中 x,y 是中心坐标，a 是宽高比，h 是高度
		// vx,vy,va,vh 是对应的速度
		_motion_mat = Eigen::MatrixXf::Identity(8, 8);
		for (int i = 0; i < ndim; i++) {
			_motion_mat(i, ndim + i) = dt;  // 位置 = 位置 + 速度 * 时间
		}
		
		// 初始化观测矩阵（4x8），将8维状态映射到4维观测空间
		_update_mat = Eigen::MatrixXf::Identity(4, 8);

		// 设置位置和速度噪声的标准差权重
		this->_std_weight_position = 1. / 20;   // 位置噪声权重
		this->_std_weight_velocity = 1. / 160;  // 速度噪声权重
	}

	// 初始化卡尔曼滤波器状态
	// 参数：measurement - 检测框 [x, y, a, h] (中心坐标，宽高比，高度)
	// 返回：初始化后的均值和协方差
	KAL_DATA ByteKalmanFilter::initiate(const DETECTBOX &measurement)
	{
		DETECTBOX mean_pos = measurement;     // 初始位置设置为检测框位置
		DETECTBOX mean_vel;
		for (int i = 0; i < 4; i++) mean_vel(i) = 0;  // 初始速度设置为0

		KAL_MEAN mean;
		// 状态均值前4维为位置，后4维为速度
		for (int i = 0; i < 8; i++) {
			if (i < 4) mean(i) = mean_pos(i);
			else mean(i) = mean_vel(i - 4);
		}

		// 计算初始协方差矩阵
		KAL_MEAN std;
		// 位置相关标准差
		std(0) = 2 * _std_weight_position * measurement[3];  // x坐标标准差
		std(1) = 2 * _std_weight_position * measurement[3];  // y坐标标准差
		std(2) = 1e-2;                                       // 宽高比标准差
		std(3) = 2 * _std_weight_position * measurement[3];  // 高度标准差
		// 速度相关标准差
		std(4) = 10 * _std_weight_velocity * measurement[3]; // x速度标准差
		std(5) = 10 * _std_weight_velocity * measurement[3]; // y速度标准差
		std(6) = 1e-5;                                       // 宽高比速度标准差
		std(7) = 10 * _std_weight_velocity * measurement[3]; // 高度速度标准差

		// 将标准差平方得到方差，构建对角协方差矩阵
		KAL_MEAN tmp = std.array().square();
		KAL_COVA var = tmp.asDiagonal();
		return std::make_pair(mean, var);
	}

	// 预测步骤：根据运动模型预测下一时刻的状态
	// 参数：mean - 状态均值（输入输出）
	//      covariance - 状态协方差（输入输出）
	void ByteKalmanFilter::predict(KAL_MEAN &mean, KAL_COVA &covariance)
	{
		// 计算过程噪声协方差
		DETECTBOX std_pos;
		std_pos << _std_weight_position * mean(3),
			_std_weight_position * mean(3),
			1e-2,
			_std_weight_position * mean(3);
		DETECTBOX std_vel;
		std_vel << _std_weight_velocity * mean(3),
			_std_weight_velocity * mean(3),
			1e-5,
			_std_weight_velocity * mean(3);
		KAL_MEAN tmp;
		tmp.block<1, 4>(0, 0) = std_pos;
		tmp.block<1, 4>(0, 4) = std_vel;
		tmp = tmp.array().square();
		KAL_COVA motion_cov = tmp.asDiagonal();
		
		// 状态预测：mean = F * mean
		KAL_MEAN mean1 = this->_motion_mat * mean.transpose();
		// 协方差预测：covariance = F * covariance * F^T + motion_cov
		KAL_COVA covariance1 = this->_motion_mat * covariance *(_motion_mat.transpose());
		covariance1 += motion_cov;

		mean = mean1;
		covariance = covariance1;
	}

	// 投影函数：将状态空间投影到观测空间
	// 参数：mean - 状态均值
	//      covariance - 状态协方差
	// 返回：投影后的观测均值和协方差
	KAL_HDATA ByteKalmanFilter::project(const KAL_MEAN &mean, const KAL_COVA &covariance)
	{
		// 观测噪声标准差
		DETECTBOX std;
		std << _std_weight_position * mean(3), _std_weight_position * mean(3),
			1e-1, _std_weight_position * mean(3);
		
		// 观测预测：mean = H * mean
		KAL_HMEAN mean1 = _update_mat * mean.transpose();
		// 协方差预测：covariance = H * covariance * H^T + R
		KAL_HCOVA covariance1 = _update_mat * covariance * (_update_mat.transpose());
		Eigen::Matrix<float, 4, 4> diag = std.asDiagonal();
		diag = diag.array().square().matrix();
		covariance1 += diag;
		return std::make_pair(mean1, covariance1);
	}

	// 更新步骤：使用检测结果更新卡尔曼滤波器状态
	// 参数：mean - 状态均值
	//      covariance - 状态协方差
	//      measurement - 检测框
	// 返回：更新后的状态均值和协方差
	KAL_DATA
		ByteKalmanFilter::update(
			const KAL_MEAN &mean,
			const KAL_COVA &covariance,
			const DETECTBOX &measurement)
	{
		// 首先将状态投影到观测空间
		KAL_HDATA pa = project(mean, covariance);
		KAL_HMEAN projected_mean = pa.first;
		KAL_HCOVA projected_cov = pa.second;

		// 计算卡尔曼增益
		// K = P * H^T * (H * P * H^T + R)^(-1)
		Eigen::Matrix<float, 4, 8> B = (covariance * (_update_mat.transpose())).transpose();
		Eigen::Matrix<float, 8, 4> kalman_gain = (projected_cov.llt().solve(B)).transpose(); // eg.8x4
		
		// 计算创新（残差）：y = z - H * x
		Eigen::Matrix<float, 1, 4> innovation = measurement - projected_mean; //eg.1x4
		
		// 状态更新：x = x + K * y
		auto tmp = innovation * (kalman_gain.transpose());
		KAL_MEAN new_mean = (mean.array() + tmp.array()).matrix();
		
		// 协方差更新：P = P - K * (H * P * H^T + R) * K^T
		KAL_COVA new_covariance = covariance - kalman_gain * projected_cov*(kalman_gain.transpose());
		return std::make_pair(new_mean, new_covariance);
	}

	// 计算门控距离（马氏距离），用于数据关联
	// 参数：mean - 状态均值
	//      covariance - 状态协方差
	//      measurements - 检测框集合
	//      only_position - 是否仅考虑位置信息
	// 返回：门控距离向量
	Eigen::Matrix<float, 1, -1>
		ByteKalmanFilter::gating_distance(
			const KAL_MEAN &mean,
			const KAL_COVA &covariance,
			const std::vector<DETECTBOX> &measurements,
			bool only_position)
	{
		// 将状态投影到观测空间
		KAL_HDATA pa = this->project(mean, covariance);
		if (only_position) {
			printf("not implement!");
			exit(0);
		}
		KAL_HMEAN mean1 = pa.first;
		KAL_HCOVA covariance1 = pa.second;

		// 计算每个检测框与预测状态之间的差异
		DETECTBOXSS d(measurements.size(), 4);
		int pos = 0;
		for (DETECTBOX box : measurements) {
			d.row(pos++) = box - mean1;
		}
		
		// 使用Cholesky分解计算马氏距离
		Eigen::Matrix<float, -1, -1, Eigen::RowMajor> factor = covariance1.llt().matrixL();
		Eigen::Matrix<float, -1, -1> z = factor.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(d).transpose();
		auto zz = ((z.array())*(z.array())).matrix();
		auto square_maha = zz.colwise().sum();
		return square_maha;
	}
}