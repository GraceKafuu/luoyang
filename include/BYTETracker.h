#pragma once

#include "STrack.h"

// 检测结果结构体
// 用于存储目标检测的结果信息
class detect_result
{
public:
    int classId;                      //< 类别ID
    float confidence;                 //< 置信度
    cv::Rect_<float> box;             //< 检测框坐标
};

// BYTETracker类
// 实现BYTE tracking算法，用于多目标跟踪
class BYTETracker
{
public:
    // 构造函数
    // param frame_rate 视频帧率，默认为30
    // param track_buffer 跟踪缓冲区大小，默认为30
    BYTETracker(int frame_rate = 30, int track_buffer = 30);
    
    // 析构函数
    ~BYTETracker();

    // 更新跟踪器状态
    // param objects 检测结果列表
    // \return 当前帧的跟踪结果
    std::vector<STrack> update(const std::vector<detect_result>& objects);
    
    // 获取指定索引的颜色
    // param idx 颜色索引
    // \return 颜色值
    cv::Scalar get_color(int idx);

private:
    // 合并两个STrack指针向量
    // param tlista 第一个STrack指针向量
    // param tlistb 第二个STrack向量
    // \return 合并后的STrack指针向量
    std::vector<STrack*> joint_stracks(std::vector<STrack*> &tlista, std::vector<STrack> &tlistb);
    
    // 合并两个STrack向量
    // param tlista 第一个STrack向量
    // param tlistb 第二个STrack向量
    // \return 合并后的STrack向量
    std::vector<STrack> joint_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);

    // 计算两个STrack向量的差集
    // param tlista 第一个STrack向量
    // param tlistb 第二个STrack向量
    // \return 差集结果
    std::vector<STrack> sub_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);
    
    // 移除重复的跟踪轨迹
    // param resa 去重后的第一个结果向量
    // param resb 去重后的第二个结果向量
    // param stracksa 第一个待去重向量
    // param stracksb 第二个待去重向量
    void remove_duplicate_stracks(std::vector<STrack> &resa, std::vector<STrack> &resb, std::vector<STrack> &stracksa, std::vector<STrack> &stracksb);

    // 线性分配算法（匈牙利算法）
    // param cost_matrix 成本矩阵
    // param cost_matrix_size 成本矩阵行数
    // param cost_matrix_size_size 成本矩阵列数
    // param thresh 阈值
    // param matches 匹配结果
    // param unmatched_a 未匹配的A类对象索引
    // param unmatched_b 未匹配的B类对象索引
    void linear_assignment(std::vector<std::vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
        std::vector<std::vector<int> > &matches, std::vector<int> &unmatched_a, std::vector<int> &unmatched_b);
        
    // 计算IOU距离（指针版本）
    // param atracks 跟踪轨迹指针向量
    // param btracks 跟踪轨迹向量
    // param dist_size 距离矩阵行数
    // param dist_size_size 距离矩阵列数
    // \return IOU距离矩阵
    std::vector<std::vector<float> > iou_distance(std::vector<STrack*> &atracks, std::vector<STrack> &btracks, int &dist_size, int &dist_size_size);
    
    // 计算IOU距离
    // param atracks 第一个跟踪轨迹向量
    // param btracks 第二个跟踪轨迹向量
    // \return IOU距离矩阵
    std::vector<std::vector<float> > iou_distance(std::vector<STrack> &atracks, std::vector<STrack> &btracks);
    
    // 计算IOU值
    // param atlbrs 第一个边界框向量
    // param btlbrs 第二个边界框向量
    // \return IOU值矩阵
    std::vector<std::vector<float> > ious(std::vector<std::vector<float> > &atlbrs, std::vector<std::vector<float> > &btlbrs);

    // LAPJV算法实现
    // param cost 成本矩阵
    // param rowsol 行解
    // param colsol 列解
    // param extend_cost 是否扩展成本矩阵
    // param cost_limit 成本限制
    // param return_cost 是否返回成本
    // \return 计算结果
    double lapjv(const std::vector<std::vector<float> > &cost, std::vector<int> &rowsol, std::vector<int> &colsol, 
        bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:
    float track_thresh;               //< 跟踪阈值
    float high_thresh;                //< 高置信度阈值
    float new_thresh;                 //< 新检测框阈值
    float match_thresh;               //< 匹配阈值
    int frame_id;                     //< 当前帧ID
    int max_time_lost;                //< 最大丢失时间

    std::vector<STrack> tracked_stracks;   //< 正在跟踪的轨迹
    std::vector<STrack> lost_stracks;      //< 丢失的轨迹
    std::vector<STrack> removed_stracks;   //< 已移除的轨迹
    byte_kalman::ByteKalmanFilter kalman_filter;  //< 卡尔曼滤波器
};