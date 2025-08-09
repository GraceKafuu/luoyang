#pragma once
#include "FaceDetect.h"
#include "PoseTransformer.h"

/**
 * @brief 人体姿态检测类
 * 
 * 该类继承自FaceDetect类，专门用于人体姿态检测任务。
 * 除了人脸检测功能外，还支持人体关键点检测及置信度计算。
 */
class PoseDetect : public FaceDetect
{
protected:
    float pointConf;                         // 关键点置信度阈值

public:
    /**
     * @brief 默认构造函数
     */
    PoseDetect(/* args */);
    
    /**
     * @brief 带参数的构造函数
     * 
     * @param dir 模型文件所在目录路径
     */
    PoseDetect(string dir);

    /**
     * @brief 执行姿态检测预测
     * 
     * 重写父类的predict方法，除了检测人体位置和人脸关键点外，
     * 还检测人体姿态关键点及其置信度信息。
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

    /**
     * @brief 获取关键点置信度阈值
     * 
     * @return float 关键点置信度阈值
     */
    float getPointConf();
};