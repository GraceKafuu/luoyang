#pragma once

#include <opencv2/opencv.hpp>
#include "BytekalmanFilter.h"

/**
 * @brief 跟踪状态枚举
 * 
 * 定义了目标跟踪的不同状态：
 * - New: 新建轨迹
 * - Tracked: 正在跟踪
 * - Lost: 跟踪丢失
 * - Removed: 轨迹移除
 */
enum TrackState { New = 0, Tracked, Lost, Removed };

/**
 * @brief 单目标跟踪类
 * 
 * STrack（Single Track）类实现了对单个目标的跟踪功能，
 * 使用卡尔曼滤波器进行状态预测和更新，维护目标的位置、
 * 尺寸和跟踪状态等信息。支持轨迹的激活、重新激活和更新操作。
 */
class STrack
{
public:
	/**
	 * @brief 构造函数
	 * 
	 * @param tlwh_ 检测框坐标 [top_left_x, top_left_y, width, height]
	 * @param score 检测置信度
	 */
	STrack( std::vector<float> tlwh_, float score);
	
	/**
	 * @brief 析构函数
	 */
	~STrack();

	/**
	 * @brief 将tlbr格式坐标转换为tlwh格式
	 * 
	 * @param tlbr tlbr格式坐标 [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
	 * @return std::vector<float> tlwh格式坐标 [top_left_x, top_left_y, width, height]
	 */
	 std::vector<float> static tlbr_to_tlwh( std::vector<float> &tlbr);
	 
	/**
	 * @brief 对多个轨迹进行批量预测
	 * 
	 * @param stracks 轨迹指针列表
	 * @param kalman_filter 卡尔曼滤波器实例
	 */
	void static multi_predict( std::vector<STrack*> &stracks, byte_kalman::ByteKalmanFilter &kalman_filter);
	
	/**
	 * @brief 计算tlwh坐标
	 */
	void static_tlwh();
	
	/**
	 * @brief 计算tlbr坐标（top-left bottom-right）
	 */
	void static_tlbr();
	
	/**
	 * @brief 将tlwh格式转换为xyah格式（center_x, center_y, aspect_ratio, height）
	 * 
	 * @param tlwh_tmp tlwh格式坐标 [top_left_x, top_left_y, width, height]
	 * @return std::vector<float> xyah格式坐标 [center_x, center_y, aspect_ratio, height]
	 */
	 std::vector<float> tlwh_to_xyah( std::vector<float> tlwh_tmp);
	 
	/**
	 * @brief 将当前tlwh格式转换为xyah格式
	 * 
	 * @return std::vector<float> xyah格式坐标 [center_x, center_y, aspect_ratio, height]
	 */
	 std::vector<float> to_xyah();
	 
	/**
	 * @brief 标记轨迹为丢失状态
	 */
	void mark_lost();
	
	/**
	 * @brief 标记轨迹为移除状态
	 */
	void mark_removed();
	
	/**
	 * @brief 获取下一个可用的轨迹ID
	 * 
	 * @return int 新的轨迹ID
	 */
	int next_id();
	
	/**
	 * @brief 获取轨迹结束帧ID
	 * 
	 * @return int 结束帧ID
	 */
	int end_frame();
	
	/**
	 * @brief 激活跟踪器
	 * 
	 * @param kalman_filter 卡尔曼滤波器实例
	 * @param frame_id 当前帧ID
	 */
	void activate(byte_kalman::ByteKalmanFilter &kalman_filter, int frame_id);
	
	/**
	 * @brief 重新激活跟踪器
	 * 
	 * @param new_track 新的跟踪对象
	 * @param frame_id 当前帧ID
	 * @param new_id 是否分配新ID，默认为false
	 */
	void re_activate(STrack &new_track, int frame_id, bool new_id = false);
	
	/**
	 * @brief 更新跟踪器
	 * 
	 * @param new_track 新的跟踪对象
	 * @param frame_id 当前帧ID
	 */
	void update(STrack &new_track, int frame_id);

public:
	bool is_activated;        // 是否已激活
	int track_id;             // 跟踪ID
	int state;                // 跟踪状态（TrackState枚举值）

	 std::vector<float> _tlwh; // 原始检测框坐标 [top_left_x, top_left_y, width, height]
	 std::vector<float> tlwh;  // 当前跟踪框坐标 [top_left_x, top_left_y, width, height]
	 std::vector<float> tlbr;  // 当前跟踪框坐标 [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
	int frame_id;             // 帧ID
	int tracklet_len;         // 跟踪轨迹长度
	int start_frame;          // 起始帧ID

	KAL_MEAN mean;            // 卡尔曼滤波器状态均值
	KAL_COVA covariance;      // 卡尔曼滤波器状态协方差
	float score;              // 检测置信度

private:
	byte_kalman::ByteKalmanFilter kalman_filter;  // 卡尔曼滤波器实例
};