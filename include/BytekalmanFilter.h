#pragma once

#include "dataType.h"

namespace byte_kalman
{
	class ByteKalmanFilter
	{
	public:
		// 卡方检验临界值表（95%置信度），用于门控距离计算
		static const double chi2inv95[10];
		
		// 构造函数
		ByteKalmanFilter();
		
		// 初始化卡尔曼滤波器状态
		// 参数：measurement - 检测框
		// 返回：初始化后的均值和协方差
		KAL_DATA initiate(const DETECTBOX& measurement);
		
		// 预测步骤：根据运动模型预测下一时刻的状态
		// 参数：mean - 状态均值（输入输出）
		//      covariance - 状态协方差（输入输出）
		void predict(KAL_MEAN& mean, KAL_COVA& covariance);
		
		// 投影函数：将状态空间投影到观测空间
		// 参数：mean - 状态均值
		//      covariance - 状态协方差
		// 返回：投影后的观测均值和协方差
		KAL_HDATA project(const KAL_MEAN& mean, const KAL_COVA& covariance);
		
		// 更新步骤：使用检测结果更新卡尔曼滤波器状态
		// 参数：mean - 状态均值
		//      covariance - 状态协方差
		//      measurement - 检测框
		// 返回：更新后的状态均值和协方差
		KAL_DATA update(const KAL_MEAN& mean,
			const KAL_COVA& covariance,
			const DETECTBOX& measurement);

		// 计算门控距离（马氏距离），用于数据关联
		// 参数：mean - 状态均值
		//      covariance - 状态协方差
		//      measurements - 检测框集合
		//      only_position - 是否仅考虑位置信息
		// 返回：门控距离向量
		Eigen::Matrix<float, 1, -1> gating_distance(
			const KAL_MEAN& mean,
			const KAL_COVA& covariance,
			const std::vector<DETECTBOX>& measurements,
			bool only_position = false);

	private:
		// 运动模型矩阵（状态转移矩阵）
		Eigen::Matrix<float, 8, 8, Eigen::RowMajor> _motion_mat;
		// 观测矩阵（将状态映射到观测值）
		Eigen::Matrix<float, 4, 8, Eigen::RowMajor> _update_mat;
		// 位置噪声标准差权重
		float _std_weight_position;
		// 速度噪声标准差权重
		float _std_weight_velocity;
	};
}
