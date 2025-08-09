#pragma once
#include "Include.h"

/**
 * @brief 图像变换类
 * 
 * 该类用于处理图像的预处理和后处理操作。
 * 主要功能包括图像的缩放、填充、归一化等预处理，
 * 以及检测结果的坐标反变换等后处理操作。
 */
class Transformer
{
protected:
    cv::Mat oriImage;           // 原始图像
    cv::Mat normalizeImage;     // 归一化后的图像
    cv::Mat inputMat;           // 输入到模型的图像矩阵

    int normalizeHeight;        // 归一化图像高度
    int normalizeWidth;         // 归一化图像宽度

    int top;                    // 上边填充像素数
    int bottom;                 // 下边填充像素数
    int left;                   // 左边填充像素数
    int right;                  // 右边填充像素数

    float resizeRatio;          // 缩放比例

public:
    /**
     * @brief 默认构造函数
     */
    Transformer(/* args */);

    /**
     * @brief 构造函数，从文件路径加载图像
     * 
     * @param imagePath 图像文件路径
     * @param normalizeHeight 目标归一化高度
     * @param normalizeWidth 目标归一化宽度
     */
    Transformer(string imagePath, int normalizeHeight, int normalizeWidth);

    /**
     * @brief 构造函数，直接使用图像矩阵
     * 
     * @param image 输入图像
     * @param normalizeHeight 目标归一化高度
     * @param normalizeWidth 目标归一化宽度
     */
    Transformer(cv::Mat image, int normalizeHeight, int normalizeWidth);

    /**
     * @brief 图像预处理
     * 
     * 对图像进行缩放、填充、归一化等操作，使其适合作为模型输入
     */
    void process();

    /**
     * @brief 坐标反变换
     * 
     * 将模型输出的坐标变换回原始图像坐标系
     * 
     * @param boxes 检测框坐标
     * @param points 关键点坐标
     */
    virtual void reverse(std::vector<cv::Rect> &boxes,
                         std::vector<std::vector<cv::Point>> &points);

    /**
     * @brief 获取归一化图像
     * 
     * @return cv::Mat 归一化后的图像
     */
    cv::Mat getNormalizeImage();

    /**
     * @brief 获取模型输入图像矩阵
     * 
     * @return cv::Mat 模型输入图像矩阵
     */
    cv::Mat getInputMat();
};