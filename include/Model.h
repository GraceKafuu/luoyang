#pragma once
#include "Include.h"

/**
 * @brief ONNX模型类
 * 
 * 该类封装了ONNX模型的加载、配置和推理功能。
 * 支持CPU和GPU(CUDA)推理，以及批量处理。
 */
class Model
{
private:
    size_t num_input_nodes;              // 输入节点数量
    vector<string> input_node_names;     // 输入节点名称列表
    vector<int64_t> input_node_dims;     // 输入节点维度信息

    int64_t input_dim_product;           // 输入维度乘积（批量大小除外）

    size_t num_output_nodes;             // 输出节点数量
    vector<string> output_node_names;    // 输出节点名称列表
    vector<int64_t> output_node_dims;    // 输出节点维度信息

    int64_t output_dim_produt;           // 输出维度乘积（批量大小除外）

    Ort::Session *ort_session;           // ONNX Runtime会话对象
    Ort::Env *env;                       // ONNX Runtime环境对象

    string onnxPath;                     // ONNX模型文件路径
    int NumThread;                       // CPU推理线程数
    string envName;                      // 环境名称

public:
    /**
     * @brief 默认构造函数
     */
    Model(/* args */);

    /**
     * @brief 析构函数，释放ONNX Runtime资源
     */
    ~Model();

    /**
     * @brief 构造函数，在CPU上运行模型
     * 
     * @param onnxPath ONNX模型文件路径
     * @param NumThread CPU推理线程数
     * @param envName 环境名称
     */
    Model(const char *onnxPath, const int NumThread, const char *envName);

    /**
     * @brief 构造函数，可选择在CPU或GPU上运行模型
     * 
     * @param onnxPath ONNX模型文件路径
     * @param NumThread CPU推理线程数
     * @param envName 环境名称
     * @param cudaId CUDA设备ID，-1表示使用CPU
     */
    Model(const char *onnxPath, const int NumThread, const char *envName, const int cudaId);

    /**
     * @brief 获取输入节点数量
     * 
     * @return const size_t 输入节点数量
     */
    const size_t getInputNum();

    /**
     * @brief 获取输入节点名称列表
     * 
     * @return const vector<string> 输入节点名称列表
     */
    const vector<string> getInputNames();

    /**
     * @brief 获取输入节点维度信息
     * 
     * @return const vector<int64_t> 输入节点维度信息
     */
    const vector<int64_t> getInputDims();

    /**
     * @brief 获取输出节点数量
     * 
     * @return const size_t 输出节点数量
     */
    const size_t getOutputNum();

    /**
     * @brief 获取输出节点名称列表
     * 
     * @return const vector<string> 输出节点名称列表
     */
    const vector<string> getOutputNames();

    /**
     * @brief 获取输出节点维度信息
     * 
     * @return const vector<int64_t> 输出节点维度信息
     */
    const vector<int64_t> getOutputNodeDims();

    /**
     * @brief 打印模型信息
     * 
     * 包括模型路径、环境名称、线程数以及输入输出节点的详细信息
     */
    void printInfo();

    /**
     * @brief 执行模型推理
     * 
     * @param images 预处理后的输入图像列表
     * @return vector<cv::Mat> 推理结果
     */
    vector<cv::Mat> predict(vector<cv::Mat> images);

    /**
     * @brief 获取输入维度乘积
     * 
     * @return int64_t 输入维度乘积（不包括批量大小维度）
     */
    int64_t getInputDimProduct();

    /**
     * @brief 获取输出维度乘积
     * 
     * @return int64_t 输出维度乘积（不包括批量大小维度）
     */
    int64_t getOutputDimProduct();
};