#include "STrack.h"

// 构造函数：根据检测框和置信度初始化STrack对象
// 参数：tlwh_ - 检测框坐标 [top_left_x, top_left_y, width, height]
//      score - 检测置信度
STrack::STrack( std::vector<float> tlwh_, float score)
{
	_tlwh.resize(4);
	_tlwh.assign(tlwh_.begin(), tlwh_.end());

	is_activated = false;     // 是否已激活
	track_id = 0;             // 跟踪ID
	state = TrackState::New;  // 跟踪状态初始化为New
	
	tlwh.resize(4);           // 存储当前跟踪框坐标 [top_left_x, top_left_y, width, height]
	tlbr.resize(4);           // 存储当前跟踪框坐标 [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

	static_tlwh();            // 计算tlwh坐标
	static_tlbr();            // 计算tlbr坐标
	frame_id = 0;             // 帧ID
	tracklet_len = 0;         // 跟踪轨迹长度
	this->score = score;      // 检测置信度
	start_frame = 0;          // 起始帧ID
}

// 析构函数
STrack::~STrack()
{
}

// 激活跟踪器
// 参数：kalman_filter - 卡尔曼滤波器实例
//      frame_id - 当前帧ID
void STrack::activate(byte_kalman::ByteKalmanFilter &kalman_filter, int frame_id)
{
	this->kalman_filter = kalman_filter;
	this->track_id = this->next_id();  // 分配新的跟踪ID

	// 将tlwh格式转换为xyah格式 [center_x, center_y, aspect_ratio, height]
	 std::vector<float> _tlwh_tmp(4);
	_tlwh_tmp[0] = this->_tlwh[0];
	_tlwh_tmp[1] = this->_tlwh[1];
	_tlwh_tmp[2] = this->_tlwh[2];
	_tlwh_tmp[3] = this->_tlwh[3];
	 std::vector<float> xyah = tlwh_to_xyah(_tlwh_tmp);
	DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];
	
	// 使用检测框初始化卡尔曼滤波器状态
	auto mc = this->kalman_filter.initiate(xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();   // 更新tlwh坐标
	static_tlbr();   // 更新tlbr坐标

	this->tracklet_len = 0;
	this->state = TrackState::Tracked;  // 设置状态为跟踪中
	if (frame_id == 1)
	{
		this->is_activated = true;
	}
	//this->is_activated = true;
	this->frame_id = frame_id;
	this->start_frame = frame_id;
}

// 重新激活跟踪器
// 参数：new_track - 新的跟踪对象
//      frame_id - 当前帧ID
//      new_id - 是否分配新ID
void STrack::re_activate(STrack &new_track, int frame_id, bool new_id)
{
	// 将新的检测框转换为xyah格式
	 std::vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
	DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];
	
	// 使用卡尔曼滤波器更新状态
	auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();  // 更新tlwh坐标
	static_tlbr();  // 更新tlbr坐标

	this->tracklet_len = 0;
	this->state = TrackState::Tracked;     // 设置状态为跟踪中
	this->is_activated = true;             // 标记为已激活
	this->frame_id = frame_id;
	this->score = new_track.score;         // 更新置信度
	if (new_id)
		this->track_id = next_id();        // 分配新ID
}

// 更新跟踪器
// 参数：new_track - 新的跟踪对象
//      frame_id - 当前帧ID
void STrack::update(STrack &new_track, int frame_id)
{
	this->frame_id = frame_id;
	this->tracklet_len++;     // 增加轨迹长度

	// 将检测框转换为xyah格式
	 std::vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
	DETECTBOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];

	// 使用卡尔曼滤波器更新状态
	auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	static_tlwh();   // 更新tlwh坐标
	static_tlbr();   // 更新tlbr坐标

	this->state = TrackState::Tracked;   // 设置状态为跟踪中
	this->is_activated = true;           // 标记为已激活

	this->score = new_track.score;       // 更新置信度
}

// 计算tlwh坐标
void STrack::static_tlwh()
{
	// 如果是新轨迹，直接使用原始检测框
	if (this->state == TrackState::New)
	{
		tlwh[0] = _tlwh[0];
		tlwh[1] = _tlwh[1];
		tlwh[2] = _tlwh[2];
		tlwh[3] = _tlwh[3];
		return;
	}

	// 否则使用卡尔曼滤波器的估计值
	tlwh[0] = mean[0];
	tlwh[1] = mean[1];
	tlwh[2] = mean[2];
	tlwh[3] = mean[3];

	// 从xyah格式转换回tlwh格式
	tlwh[2] *= tlwh[3];
	tlwh[0] -= tlwh[2] / 2;
	tlwh[1] -= tlwh[3] / 2;
}

// 计算tlbr坐标（top-left bottom-right）
void STrack::static_tlbr()
{
	tlbr.clear();
	tlbr.assign(tlwh.begin(), tlwh.end());
	tlbr[2] += tlbr[0];  // 右下角x坐标 = 左上角x坐标 + 宽度
	tlbr[3] += tlbr[1];  // 右下角y坐标 = 左上角y坐标 + 高度
}

// 将tlwh格式转换为xyah格式（center_x, center_y, aspect_ratio, height）
// 参数：tlwh_tmp - tlwh格式坐标 [top_left_x, top_left_y, width, height]
// 返回：xyah格式坐标 [center_x, center_y, aspect_ratio, height]
 std::vector<float> STrack::tlwh_to_xyah( std::vector<float> tlwh_tmp)
{
	 std::vector<float> tlwh_output = tlwh_tmp;
	tlwh_output[0] += tlwh_output[2] / 2;   // center_x = x + width/2
	tlwh_output[1] += tlwh_output[3] / 2;   // center_y = y + height/2
	tlwh_output[2] /= tlwh_output[3];       // aspect_ratio = width/height
	return tlwh_output;
}

// 将当前tlwh格式转换为xyah格式
// 返回：xyah格式坐标 [center_x, center_y, aspect_ratio, height]
 std::vector<float> STrack::to_xyah()
{
	return tlwh_to_xyah(tlwh);
}

// 将tlbr格式转换为tlwh格式
// 参数：tlbr - tlbr格式坐标 [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
// 返回：tlwh格式坐标 [top_left_x, top_left_y, width, height]
 std::vector<float> STrack::tlbr_to_tlwh( std::vector<float> &tlbr)
{
	tlbr[2] -= tlbr[0];   // width = bottom_right_x - top_left_x
	tlbr[3] -= tlbr[1];   // height = bottom_right_y - top_left_y
	return tlbr;
}

// 标记轨迹为丢失状态
void STrack::mark_lost()
{
	state = TrackState::Lost;
}

// 标记轨迹为移除状态
void STrack::mark_removed()
{
	state = TrackState::Removed;
}

// 获取下一个可用的轨迹ID
// 返回：新的轨迹ID
int STrack::next_id()
{
	static int _count = 0;
	_count++;
	return _count;
}

// 获取轨迹结束帧ID
// 返回：结束帧ID
int STrack::end_frame()
{
	return this->frame_id;
}

// 对多个轨迹进行批量预测
// 参数：stracks - 轨迹列表
//      kalman_filter - 卡尔曼滤波器实例
void STrack::multi_predict( std::vector<STrack*> &stracks, byte_kalman::ByteKalmanFilter &kalman_filter)
{
	for (int i = 0; i < stracks.size(); i++)
	{
		// 如果轨迹不是跟踪状态，将速度置零
		if (stracks[i]->state != TrackState::Tracked)
		{
			stracks[i]->mean[7] = 0;
		}
		// 使用卡尔曼滤波器进行预测
		kalman_filter.predict(stracks[i]->mean, stracks[i]->covariance);
	}
}