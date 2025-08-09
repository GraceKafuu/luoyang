#include "PoseTransformer.h"


/**
 * @brief 默认构造函数
 */
PoseTransformer::PoseTransformer()
{
}

/**
 * @brief 构造函数，直接使用图像矩阵
 * 
 * @param image 输入图像
 * @param normalizeHeight 目标归一化高度
 * @param normalizeWidth 目标归一化宽度
 * @param pointNum 人体姿态关键点数量
 */
PoseTransformer::PoseTransformer(cv::Mat image,
                                 int normalizeHeight,
                                 int normalizeWidth,
                                 int pointNum)
{
    this->pointNum = pointNum;
    this->oriImage = image;
    this->normalizeHeight = normalizeHeight;
    this->normalizeWidth = normalizeWidth;
}

/**
 * @brief 构造函数，从文件路径加载图像
 * 
 * @param imagePath 图像文件路径
 * @param normalizeHeight 目标归一化高度
 * @param normalizeWidth 目标归一化宽度
 * @param pointNum 人体姿态关键点数量
 */
PoseTransformer::PoseTransformer(string imagePath,
                                 int normalizeHeight,
                                 int normalizeWidth,
                                 int pointNum)
{
    this->pointNum = pointNum;
    this->oriImage = cv::imread(imagePath);
    this->normalizeHeight = normalizeHeight;
    this->normalizeWidth = normalizeWidth;
}