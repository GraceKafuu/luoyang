#pragma once
#include "Include.h"

class Model
{
private:
    // input node number
    size_t num_input_nodes;
    // input node names
    vector<string> input_node_names;
    // input node dims
    vector<int64_t> input_node_dims;

    int64_t input_dim_product;

    // output node number
    size_t num_output_nodes;
    // output node names
    vector<string> output_node_names;
    // output node dims
    vector<int64_t> output_node_dims;

    int64_t output_dim_produt;

    Ort::Session *ort_session;

    Ort::Env *env;

    string onnxPath;

    int NumThread;

    string envName;

public:
    Model(/* args */);

    ~Model();

    Model(const char *onnxPath, const int NumThread, const char *envName);

    Model(const char *onnxPath, const int NumThread, const char *envName, const int cudaId);

    const size_t getInputNum();

    const vector<string> getInputNames();

    const vector<int64_t> getInputDims();

    const size_t getOutputNum();

    const vector<string> getOutputNames();

    const vector<int64_t> getOutputNodeDims();

    void printInfo();

    vector<cv::Mat> predict(vector<cv::Mat> images);

    int64_t getInputDimProduct();

    int64_t getOutputDimProduct();
};