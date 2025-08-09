#pragma once
#include "Detect.h"
#include "FaceTransformer.h"

/**
 * @brief 人脸检测类
 * 
 * 该类继承自Detect类，专门用于人脸检测任务。
 * 除了基本的目标检测功能外，还支持人脸关键点检测。
 */
class FaceDetect : public Detect
{
protected:
    int pointNum;                            // 人脸关键点数量

public:
    /**
     * @brief 默认构造函数
     */
    FaceDetect();
    
    /**
     * @brief 带参数的构造函数
     * 
     * @param dir 模型文件所在目录路径
     */
    FaceDetect(string dir);

    /**
     * @brief 执行人脸检测预测
     * 
     * 重写父类的predict方法，除了检测人脸位置外，
     * 还检测人脸关键点信息。
     * 
     * @param images 输入图像列表
     * @param outputRects 输出检测框列表
     * @param outputNames 输出类别名称列表
     * @param outputConfidences 输出置信度列表
     * @param outputPoints 输出关键点列表
     * @param outputPointConfidences 输出关键点置信度列表
     */
    virtual void predict(vector<cv::Mat> images,
                         std::vector<std::vector<cv::Rect>> &outputRects,
                         std::vector<std::vector<string>> &outputNames,
                         std::vector<std::vector<float>> &outputConfidences,
                         std::vector<std::vector<std::vector<cv::Point>>> &outputPoints,
                         std::vector<std::vector<std::vector<float>>> &outputPointConfidences);
};