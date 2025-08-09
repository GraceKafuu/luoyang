#pragma once
#include "Include.h"
#include "Model.h"
#include "Transformer.h"

/**
 * @brief 目标检测基类
 * 
 * 该类封装了目标检测的基本功能，包括模型加载、图像预处理、
 * 模型推理和后处理等操作。作为基类，可以被其他特定检测类继承。
 */
class Detect
{
private:
public:
    /**
     * @brief 默认构造函数
     */
    Detect(/* args */);

    /**
     * @brief 带参数的构造函数
     * 
     * @param dir 模型文件所在目录路径
     */
    Detect(string dir);

    /**
     * @brief 析构函数，负责释放模型资源
     */
    ~Detect();

    /**
     * @brief 获取批处理大小
     * 
     * @return int 批处理大小
     */
    int getBatchSize();

    /**
     * @brief 获取类别数量
     * 
     * @return int 类别数量
     */
    int getClassNum();

    /**
     * @brief 获取非极大值抑制置信度阈值
     * 
     * @return float NMS置信度阈值
     */
    float getNmsConf();

    /**
     * @brief 获取目标置信度阈值
     * 
     * @return float 目标置信度阈值
     */
    float getObjConf();

    /**
     * @brief 执行目标检测预测
     * 
     * @param images 输入图像列表
     * @param outputRects 输出检测框列表
     * @param outputNames 输出类别名称列表
     * @param outputConfidences 输出置信度列表
     * @param outputPoints 输出关键点列表
     * @param outputPointConfidences 输出关键点置信度列表
     */
    virtual void predict(vector<cv::Mat> images,
                         vector<vector<cv::Rect>> &outputRects,
                         vector<vector<string>> &outputNames,
                         vector<vector<float>> &outputConfidences,
                         vector<vector<vector<cv::Point>>> &outputPoints,
                         vector<vector<vector<float>>> &outputPointConfidences);

    /**
     * @brief 预热模型，执行一次推理以初始化模型
     */
    virtual void warmup();

protected:
    int batchSize;                            //批处理大小
    vector<string> classNames;                // 类别名称列表
    Model *model;                             // 模型指针
    float nmsConf;                            // 非极大值抑制置信度阈值
    float objConf;                            // 目标置信度阈值
    int deviceId;                             // 设备ID
    bool useNms;                              // 是否使用非极大值抑制

};