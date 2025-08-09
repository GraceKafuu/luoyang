#pragma once
#include "Transformer.h"

/**
 * @brief 人脸图像变换类
 * 
 * 该类继承自Transformer类，专门用于人脸检测相关的图像变换操作。
 * 除了基本的图像预处理和后处理功能外，还支持人脸关键点的坐标变换。
 */
class FaceTransformer : public Transformer
{

protected:
    int pointNum;               // 人脸关键点数量

public:
    /**
     * @brief 默认构造函数
     */
    FaceTransformer();

    /**
     * @brief 构造函数，从文件路径加载图像
     * 
     * @param imagePath 图像文件路径
     * @param normalizeHeight 目标归一化高度
     * @param normalizeWidth 目标归一化宽度
     * @param pointNum 人脸关键点数量
     */
    FaceTransformer(string imagePath,
                    int normalizeHeight,
                    int normalizeWidth,
                    int pointNum);

    /**
     * @brief 构造函数，直接使用图像矩阵
     * 
     * @param image 输入图像
     * @param normalizeHeight 目标归一化高度
     * @param normalizeWidth 目标归一化宽度
     * @param pointNum 人脸关键点数量
     */
    FaceTransformer(cv::Mat image,
                    int normalizeHeight,
                    int normalizeWidth,
                    int pointNum);

    /**
     * @brief 坐标反变换
     * 
     * 重写父类的reverse方法，除了对检测框进行坐标反变换外，
     * 还对人脸关键点坐标进行反变换。
     * 
     * @param boxes 检测框坐标
     * @param points 人脸关键点坐标
     */
    virtual void reverse(std::vector<cv::Rect> &boxes, std::vector<std::vector<cv::Point>> &points);
};