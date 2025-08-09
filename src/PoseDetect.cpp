#include "PoseDetect.h"


/**
 * @brief 默认构造函数
 */
PoseDetect::PoseDetect(/* args */)
{
}

/**
 * @brief 带参数的构造函数，初始化姿态检测器
 * 
 * 从指定目录加载模型文件、类别名称文件和参数配置文件，
 * 并根据配置参数创建模型实例。与人脸检测器不同的是，
 * 还需要读取关键点置信度阈值参数。
 * 
 * @param dir 模型文件所在目录路径
 */
PoseDetect::PoseDetect(string dir)
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
    this->pointNum = stoi(paramMap["point_num"]);
    int NumThread = stoi(paramMap["num_thread"]);
    this->pointConf = stof(paramMap["point_conf"]);

    this->useNms = paramMap.count("need_nms") && stoi(paramMap["need_nms"]) == 0 ?  false : true;



    string envName = "yolo";
    this->model = new Model(onnxPath.c_str(), NumThread, envName.c_str(), this->deviceId);
    this->model->printInfo();
}

/**
 * @brief 执行姿态检测预测
 * 
 * 对输入图像进行预处理，使用模型进行推理，并对结果进行后处理，
 * 包括置信度过滤、非极大值抑制和人体关键点处理等操作。
 * 与人脸检测器不同的是，还会处理关键点置信度信息。
 * 
 * @param images 输入图像列表
 * @param outputRects 输出检测框列表
 * @param outputNames 输出类别名称列表
 * @param outputConfidences 输出置信度列表
 * @param outputPoints 输出关键点列表
 * @param outputPointConfidences 输出关键点置信度列表
 */
void PoseDetect::predict(vector<cv::Mat> images,
                         std::vector<std::vector<cv::Rect>> &outputRects,
                         std::vector<std::vector<string>> &outputNames,
                         std::vector<std::vector<float>> &outputConfidences,
                         std::vector<std::vector<std::vector<cv::Point>>> &outputPoints,
                         std::vector<std::vector<std::vector<float>>> &outputPointConfidences)
{
    // 创建姿态图像变换器和预处理后的图像列表
    vector<PoseTransformer> poseTransformers;
    poseTransformers.reserve(images.size());
    vector<cv::Mat> inputImages;
    inputImages.reserve(images.size());

    // 对每张图像进行预处理，使用姿态专用变换器
    for (cv::Mat image : images)
    {
        PoseTransformer transformer(image, this->model->getInputDims().at(2), this->model->getInputDims().at(3), this->pointNum);
        transformer.process();
        inputImages.push_back(transformer.getInputMat());
        poseTransformers.push_back(transformer);
    }

    // 使用模型进行推理
    std::vector<cv::Mat> predicts = this->model->predict(inputImages);

    // 处理每个推理结果
    for (int i = 0; i < predicts.size(); i++)
    {
        cv::Mat predict = predicts[i];

        std::vector<cv::Rect> boxes;
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<std::vector<cv::Point>> points;
        std::vector<std::vector<float>> pointConfidences;

        // 解析模型输出，提取检测框、类别、置信度、人体关键点和关键点置信度
        for (int i = 0; i < predict.rows; i++)
        {
            // 获取目标置信度
            float conf = predict.at<float>(i, 4);
            if (conf < this->objConf)
            {
                continue;
            }
            
            // 获取类别置信度
            cv::Mat classScores = predict.row(i).colRange(5 + 3 * this->pointNum, 5 + 3 * this->pointNum + this->classNames.size());

            cv::Point classIdPoint;
            double clsConf;
            cv::minMaxLoc(classScores, 0, &clsConf, 0, &classIdPoint);

            // 综合置信度过滤
            if (conf * clsConf < this->objConf)
            {
                continue;
            }

            // 提取检测框坐标
            float cx = predict.at<float>(i, 0);
            float cy = predict.at<float>(i, 1);
            float w = predict.at<float>(i, 2);
            float h = predict.at<float>(i, 3);

            float left = cx - 0.5f * w;
            float top = cy - 0.5f * h;
            cv::Rect box(left, top, w, h);

            boxes.push_back(box);
            classIds.push_back(classIdPoint.x);
            confidences.push_back(conf * static_cast<float>(clsConf));

            // 提取人体关键点坐标和关键点置信度
            std::vector<cv::Point> localPoints;
            localPoints.reserve(this->pointNum);

            std::vector<float> localPointConfidence;
            localPointConfidence.reserve(this->pointNum);

            for (int point_id = 0; point_id < this->pointNum; point_id++)
            {
                float pointX = predict.at<float>(i, 5 + 2 * point_id);
                float pointY = predict.at<float>(i, 5 + 2 * point_id + 1);
                cv::Point point = cv::Point(pointX, pointY);
                localPoints.push_back(point);

                localPointConfidence.push_back(predict.at<float>(i, 5 + 2 * this->pointNum + point_id));
            }

            points.push_back(localPoints);
            pointConfidences.push_back(localPointConfidence);
        }
        
        // 根据是否使用NMS进行不同处理
        if(this->useNms){
            std::vector<int> indexes;
            cv::dnn::NMSBoxesBatched(boxes, confidences, classIds, this->objConf, this->nmsConf, indexes);
            std::vector<cv::Rect> outputRect;
            std::vector<float> outputConfidence;
            std::vector<string> outputName;
            std::vector<std::vector<cv::Point>> outputPoint;
            std::vector<std::vector<float>> outputPointConfidence;

            outputRect.reserve(indexes.size());
            outputConfidence.reserve(indexes.size());
            outputName.reserve(indexes.size());
            outputPoints.reserve(indexes.size());
            outputPointConfidence.reserve(indexes.size());

            // 根据NMS结果提取最终检测结果、关键点和关键点置信度
            for (int index : indexes)
            {
                outputRect.push_back(boxes.at(index));
                outputConfidence.push_back(confidences.at(index));
                outputName.push_back(this->classNames[classIds.at(index)]);
                outputPoint.push_back(points.at(index));
                outputPointConfidence.push_back(pointConfidences.at(index));
            }
            
            // 对检测框和关键点进行坐标反变换
            poseTransformers[i].reverse(outputRect, outputPoint);
            outputRects.push_back(outputRect);
            outputConfidences.push_back(outputConfidence);
            outputNames.push_back(outputName);
            outputPoints.push_back(outputPoint);
            outputPointConfidences.push_back(outputPointConfidence);
        }else{
            vector<string> outputName;
            outputName.reserve(classIds.size());

            for (int classId: classIds){
                outputName.push_back(this->classNames[classId]);
            }    
            
            // 对检测框和关键点进行坐标反变换        
            poseTransformers[i].reverse(boxes, points);
            outputRects.push_back(boxes);
            outputConfidences.push_back(confidences); 
            outputNames.push_back(outputName);
            outputPoints.push_back(points);
            outputPointConfidences.push_back(pointConfidences);
        }
        
    }
}

/**
 * @brief 获取关键点置信度阈值
 * 
 * @return float 关键点置信度阈值
 */
float PoseDetect::getPointConf()
{
    return this->pointConf;
}