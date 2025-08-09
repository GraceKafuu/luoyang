#pragma once
#include "FaceTransformer.h"

/**
 * @brief 姿态图像变换类
 * 
 * 该类继承自FaceTransformer类，专门用于人体姿态检测相关的图像变换操作。
 * 由于人体姿态检测使用与人脸检测相同的关键点处理逻辑，
 * 所以该类直接继承FaceTransformer而无需重写方法。
 */
class PoseTransformer : public FaceTransformer
{
public:
    /**
     * @brief 默认构造函数
     */
    PoseTransformer();

    /**
     * @brief 构造函数，从文件路径加载图像
     * 
     * @param imagePath 图像文件路径
     * @param normalizeHeight 目标归一化高度
     * @param normalizeWidth 目标归一化宽度
     * @param pointNum 人体姿态关键点数量
     */
    PoseTransformer(string imagePath,
                    int normalizeHeight,
                    int normalizeWidth,
                    int pointNum);

    /**
     * @brief 构造函数，直接使用图像矩阵
     * 
     * @param image 输入图像
     * @param normalizeHeight 目标归一化高度
     * @param normalizeWidth 目标归一化宽度
     * @param pointNum 人体姿态关键点数量
     */
    PoseTransformer(cv::Mat image,
                    int normalizeHeight,
                    int normalizeWidth,
                    int pointNum);
};