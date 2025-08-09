#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

// left_x, top_y, w, h
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;

typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;
// 状态向量 [x,y,w,h,vx,vy,vw,vh]  (位置 + 速度)
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
// 状态协方差矩阵 表示状态的不确定性
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
// 观测向量 [x,y,w,h]
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
// 观测噪声协方差矩阵 观测误差的不确定性
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;

using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;

using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

using RESULT_DATA = std::pair<int, DETECTBOX>;

typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM;
