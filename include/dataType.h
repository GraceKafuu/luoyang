#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

// 特征提取模型路径
const std::string  k_feature_model_path ="./feature.onnx";
// 目标检测模型路径
const std::string  k_detect_model_path ="./yolov5s.onnx";

// 检测框类型定义：单个检测框 (1x4矩阵，格式为 [x1, y1, x2, y2])
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;
// 多个检测框集合类型定义：多个检测框组成的矩阵 (nx4矩阵)
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;

// 卡尔曼滤波相关类型定义
// 卡尔曼滤波器状态均值 (1x8矩阵)
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
// 卡尔曼滤波器状态协方差 (8x8矩阵)
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
// 卡尔曼滤波器观测值均值 (1x4矩阵)
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
// 卡尔曼滤波器观测值协方差 (4x4矩阵)
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;
// 卡尔曼滤波器状态数据对：均值和协方差
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
// 卡尔曼滤波器观测数据对：均值和协方差
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

// 主程序使用的结果数据类型：跟踪ID和检测框的配对
using RESULT_DATA = std::pair<int, DETECTBOX>;

// 匹配数据：跟踪ID和检测ID的配对
using MATCH_DATA = std::pair<int, int>;
// 跟踪匹配结果结构体：包含匹配对、未匹配的跟踪器和未匹配的检测
typedef struct t{
    std::vector<MATCH_DATA> matches;           // 匹配成功的跟踪ID-检测ID对
    std::vector<int> unmatched_tracks;        // 未匹配的跟踪器ID列表
    std::vector<int> unmatched_detections;    // 未匹配的检测ID列表
}TRACHER_MATCHD;

// 动态矩阵类型定义：用于线性分配算法中的成本矩阵
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM;