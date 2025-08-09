#include "Detect.h"

/**
 * @brief 默认构造函数
 */
Detect::Detect(/* args */)
{
}

/**
 * @brief 析构函数，释放模型资源
 */
Detect::~Detect()
{
    std::cout << "模型释放资源 " << std::endl;
    delete this->model;
}

/**
 * @brief 带参数的构造函数，初始化检测器
 * 
 * 从指定目录加载模型文件、类别名称文件和参数配置文件，
 * 并根据配置参数创建模型实例。
 * 
 * @param dir 模型文件所在目录路径
 */
Detect::Detect(const string dir)
{

    string classPath = dir + "/names.txt";
    string paramPath = dir + "/param.map";
    string onnxPath = dir + "/yolo.onnx";

    this->classNames = readLines(classPath);

    unordered_map<string, string> paramMap = readMap(paramPath);
    this->batchSize = stoi(paramMap["batch_size"]);
    this->nmsConf = stof(paramMap["nms_conf"]);
    this->objConf = stof(paramMap["conf_threshold"]);
    this->deviceId = stoi(paramMap["device_id"]);
    this->useNms = paramMap.count("need_nms") && stoi(paramMap["need_nms"]) == 0 ?  false : true;
    int NumThread = stoi(paramMap["num_thread"]);

    string envName = "yolo";
    this->model = new Model(onnxPath.c_str(), NumThread, envName.c_str(), this->deviceId);
    this->model->printInfo();
}

/**
 * @brief 获取非极大值抑制置信度阈值
 * 
 * @return float NMS置信度阈值
 */
float Detect::getNmsConf()
{
    return this->nmsConf;
}

/**
 * @brief 获取目标置信度阈值
 * 
 * @return float 目标置信度阈值
 */
float Detect::getObjConf()
{
    return this->objConf;
}

/**
 * @brief 获取批处理大小
 * 
 * @return int 批处理大小
 */
int Detect::getBatchSize()
{
    return this->batchSize;
}

/**
 * @brief 获取类别数量
 * 
 * @return int 类别数量
 */
int Detect::getClassNum()
{
    return this->classNames.size();
}

/**
 * @brief 执行目标检测预测
 * 
 * 对输入图像进行预处理，使用模型进行推理，并对结果进行后处理，
 * 包括置信度过滤和非极大值抑制等操作。
 * 
 * @param images 输入图像列表
 * @param outputRects 输出检测框列表
 * @param outputNames 输出类别名称列表
 * @param outputConfidences 输出置信度列表
 * @param outputPoints 输出关键点列表
 * @param outputPointConfidences 输出关键点置信度列表
 */
void Detect::predict(vector<cv::Mat> images,
                     vector<vector<cv::Rect>> &outputRects,
                     vector<vector<string>> &outputNames,
                     vector<vector<float>> &outputConfidences,
                     vector<vector<vector<cv::Point>>> &outputPoints,
                     vector<vector<vector<float>>> &outputPointConfidences)
{
    // 创建图像变换器和预处理后的图像列表
    vector<Transformer> transformers;
    transformers.reserve(images.size());
    vector<cv::Mat> inputImages;
    inputImages.reserve(images.size());

    // 对每张图像进行预处理
    for (cv::Mat image : images)
    {
        Transformer transformer(image, this->model->getInputDims().at(2), this->model->getInputDims().at(3));
        transformer.process();
        inputImages.push_back(transformer.getInputMat());
        transformers.push_back(transformer);
    }

    // 使用模型进行推理
    vector<cv::Mat> predicts = this->model->predict(inputImages);

    // 处理每个推理结果
    for (int i = 0; i < predicts.size(); i++)
    {
        cv::Mat predict = predicts[i];

        vector<cv::Rect> boxes;
        vector<int> classIds;
        vector<float> confidences;

        // 解析模型输出，提取检测框、类别和置信度
        for (int i = 0; i < predict.rows; i++)
        {
            float conf = predict.at<float>(i, 4);
            if (conf < this->objConf)
            {
                continue;
            }
            cv::Mat classScores = predict.row(i).colRange(5, 5 + this->classNames.size());

            cv::Point classIdPoint;
            double clsConf;
            cv::minMaxLoc(classScores, 0, &clsConf, 0, &classIdPoint);

            float cx = predict.at<float>(i, 0);
            float cy = predict.at<float>(i, 1);
            float w = predict.at<float>(i, 2);
            float h = predict.at<float>(i, 3);

            float left = cx - 0.5f * w;
            float top = cy - 0.5f * h;
            cv::Rect box(left, top, w, h);

            boxes.push_back(box);
            classIds.push_back(classIdPoint.x);
            confidences.push_back(conf * clsConf);
        }

        // 根据是否使用NMS进行不同处理
        if (this->useNms){
            vector<int> indexes;
            cv::dnn::NMSBoxesBatched(boxes, confidences, classIds, this->objConf, this->nmsConf, indexes);

            vector<cv::Rect> outputRect;
            vector<float> outputConfidence;
            vector<string> outputName;

            outputRect.reserve(indexes.size());
            outputConfidence.reserve(indexes.size());
            outputName.reserve(indexes.size());

            // 根据NMS结果提取最终检测结果
            for (int index : indexes)
            {
                outputRect.push_back(boxes.at(index));
                outputConfidence.push_back(confidences.at(index));
                outputName.push_back(this->classNames[classIds.at(index)]);
            }
            vector<vector<cv::Point>> points;
            transformers[i].reverse(outputRect, points);
            outputRects.push_back(outputRect);
            outputConfidences.push_back(outputConfidence);
            outputNames.push_back(outputName);
        }else{
            vector<string> outputName;
            outputName.reserve(classIds.size());

            for (int classId: classIds){
                outputName.push_back(this->classNames[classId]);
            }

            vector<vector<cv::Point>> points;
            transformers[i].reverse(boxes, points);
            outputRects.push_back(boxes);
            outputConfidences.push_back(confidences); 
            outputNames.push_back(outputName);
        }
    }
}

/**
 * @brief 预热模型，执行一次推理以初始化模型
 * 
 * 创建一批黑色图像作为输入，执行一次完整的推理过程，
 * 以确保模型在首次推理时的性能表现。
 */
void Detect::warmup()
{
    cv::Mat image(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<std::vector<cv::Rect>> outputRects;
    std::vector<std::vector<string>> outputNames;
    std::vector<std::vector<float>> outputConfidences;
    std::vector<std::vector<std::vector<cv::Point>>> points;
    std::vector<std::vector<std::vector<float>>> pointConfidences;
    vector<cv::Mat> images;
    for (int i = 0; i < this->batchSize; i++)
        images.push_back(image.clone());

    this->predict(images,
                  outputRects,
                  outputNames,
                  outputConfidences,
                  points,
                  pointConfidences);
}