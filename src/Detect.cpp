#include "Detect.h"

Detect::Detect(/* args */)
{
}

Detect::~Detect()
{
    std::cout << "模型释放资源 " << std::endl;
    delete this->model;
}

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

float Detect::getNmsConf()
{
    return this->nmsConf;
}
float Detect::getObjConf()
{
    return this->objConf;
}
int Detect::getBatchSize()
{
    return this->batchSize;
}

int Detect::getClassNum()
{
    return this->classNames.size();
}
void Detect::predict(vector<cv::Mat> images,
                     vector<vector<cv::Rect>> &outputRects,
                     vector<vector<string>> &outputNames,
                     vector<vector<float>> &outputConfidences,
                     vector<vector<vector<cv::Point>>> &outputPoints,
                     vector<vector<vector<float>>> &outputPointConfidences)
{
    vector<Transformer> transformers;
    transformers.reserve(images.size());
    vector<cv::Mat> inputImages;
    inputImages.reserve(images.size());

    for (cv::Mat image : images)
    {
        Transformer transformer(image, this->model->getInputDims().at(2), this->model->getInputDims().at(3));
        transformer.process();
        inputImages.push_back(transformer.getInputMat());
        transformers.push_back(transformer);
    }

    vector<cv::Mat> predicts = this->model->predict(inputImages);

    for (int i = 0; i < predicts.size(); i++)
    {
        cv::Mat predict = predicts[i];

        vector<cv::Rect> boxes;
        vector<int> classIds;
        vector<float> confidences;

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

        if (this->useNms){
            vector<int> indexes;
            cv::dnn::NMSBoxesBatched(boxes, confidences, classIds, this->objConf, this->nmsConf, indexes);

            vector<cv::Rect> outputRect;
            vector<float> outputConfidence;
            vector<string> outputName;

            outputRect.reserve(indexes.size());
            outputConfidence.reserve(indexes.size());
            outputName.reserve(indexes.size());

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