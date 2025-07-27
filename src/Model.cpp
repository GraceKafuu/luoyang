#include "Model.h"

Model::Model(/* args */)
{
}

Model::~Model()
{
    delete this->ort_session;
    delete this->env;
}

Model::Model(const char *onnxPath, const int NumThread, const char *envName)
{
    Model(onnxPath, NumThread, envName, -1);
}

Model::Model(const char *onnxPath, const int NumThread, const char *envName, const int cudaId)
{
    this->onnxPath = onnxPath;
    this->NumThread = NumThread;
    this->envName = envName;

    Ort::SessionOptions sessionOptions;

    std::vector<std::string> avaiableProviders = Ort::GetAvailableProviders();

    auto cudaAvailable = std::find(avaiableProviders.begin(), avaiableProviders.end(), "CUDAExecutionProvider");
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (cudaId < 0 || cudaAvailable == avaiableProviders.end())
    {
        std::cout << "Your ORT build without GPU. Changle to CPU" << std::endl;
        std::cout << "Infer model on CPU" << std::endl;
        sessionOptions.SetIntraOpNumThreads(NumThread);
    }
    else
    {
        std::cout << "Infer model on GPU" << std::endl;
        OrtCUDAProviderOptions cudaOption;
        cudaOption.device_id = cudaId;
        cudaOption.arena_extend_strategy = 0;
        cudaOption.gpu_mem_limit = 4 * 1024 * 1024 * 1024LL;  // 4GB
        cudaOption.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cudaOption.do_copy_in_default_stream = 1;
        cudaOption.has_user_compute_stream = 0;
        cudaOption.user_compute_stream = nullptr;
        cudaOption.default_memory_arena_cfg = nullptr;

        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }

    this->env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, envName);

    this->ort_session = new Ort::Session(*this->env, onnxPath, sessionOptions);

    this->num_input_nodes = this->ort_session->GetInputCount();
    vector<Ort::AllocatedStringPtr> input_names_ptr;
    input_names_ptr.reserve(this->num_input_nodes);
    this->input_node_names.reserve(this->num_input_nodes);
    for (size_t i = 0; i < this->num_input_nodes; i++)
    {
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = this->ort_session->GetInputNameAllocated(i, allocator);
        this->input_node_names.push_back(input_name.get());
        auto input_type_info = this->ort_session->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        this->input_node_dims = input_tensor_info.GetShape();
    }

    this->num_output_nodes = this->ort_session->GetOutputCount();
    this->output_node_names.reserve(this->num_output_nodes);
    for (size_t i = 0; i < this->num_output_nodes; i++)
    {
        Ort::AllocatorWithDefaultOptions allocator;
        auto output_name = this->ort_session->GetOutputNameAllocated(i, allocator);
        this->output_node_names.push_back(output_name.get());
        auto output_type_info = this->ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        this->output_node_dims = output_tensor_info.GetShape();
    }

    this->input_dim_product = 1L;
    for (int64_t d = 1; d < this->input_node_dims.size(); d++)
    {
        this->input_dim_product = this->input_dim_product * this->input_node_dims.at(d);
    }
    this->output_dim_produt = 1L;
    for (int64_t d = 1; d < this->output_node_dims.size(); d++)
    {
        this->output_dim_produt = this->output_dim_produt * this->output_node_dims.at(d);
    }
}
const size_t Model::getInputNum()
{
    return this->num_input_nodes;
}

const vector<string> Model::getInputNames()
{
    return this->input_node_names;
}

const vector<int64_t> Model::getInputDims()
{
    return this->input_node_dims;
}

const size_t Model::getOutputNum()
{
    return this->num_output_nodes;
}

const vector<string> Model::getOutputNames()
{
    return this->output_node_names;
}

const vector<int64_t> Model::getOutputNodeDims()
{
    return this->output_node_dims;
}

void Model::printInfo()
{
    cout << "onnxPath = " << this->onnxPath << endl;
    cout << "envName = " << this->envName << endl;
    cout << "NumThread = " << this->NumThread << endl;

    cout << "Number of inputs = " << this->num_input_nodes << endl;
    for (size_t i = 0; i < num_input_nodes; i++)
    {
        cout << "Input " << i << " : name =" << this->input_node_names[i] << endl;
        cout << "Input " << i << " : num_dims = " << this->input_node_dims.size() << '\n';
        for (size_t j = 0; j < this->input_node_dims.size(); j++)
        {
            cout << "Input " << i << " : dim[" << j << "] =" << this->input_node_dims[j] << '\n';
        }
        cout << flush;
    }

    cout << "Number of outputs = " << this->num_output_nodes << endl;
    for (size_t i = 0; i < num_output_nodes; i++)
    {
        cout << "Output " << i << " : name =" << this->output_node_names[i] << endl;
        cout << "Output " << i << " : num_dims = " << this->output_node_dims.size() << '\n';
        for (size_t j = 0; j < this->output_node_dims.size(); j++)
        {
            cout << "Output " << i << " : dim[" << j << "] =" << this->output_node_dims[j] << '\n';
        }
        cout << flush;
    }
}

vector<cv::Mat> Model::predict(vector<cv::Mat> images)
{
    int64_t batch_size = images.size();

    size_t inputTensorSize = this->input_dim_product * batch_size;
    vector<float> inputTensorValues(inputTensorSize);

    size_t outputTensorSize = this->output_dim_produt * batch_size;
    vector<float> outputTensorValues(outputTensorSize);

    // Copy image processed to InputTensorValues
    for (int i = 0; i < batch_size; ++i)
    {
        copy(images[i].begin<float>(),
             images[i].end<float>(),
             inputTensorValues.begin() + i * inputTensorSize / batch_size);
    }

    vector<Ort::Value> inputTensors;
    vector<Ort::Value> outputTensors;

    vector<int64_t> inputShape = this->input_node_dims;
    inputShape.at(0) = batch_size;

    vector<int64_t> outputShape = this->output_node_dims;
    outputShape.at(0) = batch_size;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo,
                                                           inputTensorValues.data(),
                                                           inputTensorSize,
                                                           inputShape.data(),
                                                           inputShape.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo,
                                                            outputTensorValues.data(),
                                                            outputTensorSize,
                                                            outputShape.data(),
                                                            outputShape.size()));

    vector<const char *> inputNames(this->input_node_names.size(), nullptr);
    for (int i = 0; i < this->input_node_names.size(); i++)
    {
        inputNames[i] = this->input_node_names[i].c_str();
    }

    vector<const char *> outputNames(this->output_node_names.size(), nullptr);
    for (int i = 0; i < this->output_node_names.size(); i++)
    {
        outputNames[i] = this->output_node_names[i].c_str();
    }

    this->ort_session->Run(Ort::RunOptions{nullptr},
                           inputNames.data(),
                           inputTensors.data(),
                           this->num_input_nodes,
                           outputNames.data(),
                           outputTensors.data(),
                           this->num_output_nodes);

    vector<cv::Mat> predicts;
    for (int batch_id = 0; batch_id < batch_size; batch_id++)
    {
        vector<float> predict(this->output_dim_produt);
        copy(outputTensorValues.data() + batch_id * this->output_dim_produt,
             outputTensorValues.data() + (batch_id + 1) * this->output_dim_produt,
             predict.begin());

        cv::Mat predictMat(this->output_node_dims[1], this->output_node_dims[2], CV_32F, predict.data());
        predicts.push_back(predictMat);
    }
    return predicts;
}

int64_t Model::getInputDimProduct()
{
    return this->input_dim_product;
}

int64_t Model::getOutputDimProduct()
{
    return this->output_dim_produt;
}