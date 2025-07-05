#include "FaceDetect.h"

FaceDetect::FaceDetect(/* args */)
{
}

FaceDetect::FaceDetect(string dir)
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

    string envName = "yolo";
    this->model = new Model(onnxPath.c_str(), NumThread, envName.c_str(), this->deviceId);
    this->model->printInfo();
}

void FaceDetect::predict(vector<cv::Mat> images,
                         std::vector<std::vector<cv::Rect>> &outputRects,
                         std::vector<std::vector<string>> &outputNames,
                         std::vector<std::vector<float>> &outputConfidences,
                         std::vector<std::vector<std::vector<cv::Point>>> &outputPoints,
                         std::vector<std::vector<std::vector<float>>> &outputPointConfidences)
{
    vector<FaceTransformer> faceTransformers;
    faceTransformers.reserve(images.size());
    vector<cv::Mat> inputImages;
    inputImages.reserve(images.size());

    for (cv::Mat image : images)
    {
        FaceTransformer transformer(image, this->model->getInputDims().at(2), this->model->getInputDims().at(3), this->pointNum);
        transformer.process();
        inputImages.push_back(transformer.getInputMat());
        faceTransformers.push_back(transformer);
    }

    std::vector<cv::Mat> predicts = this->model->predict(inputImages);

    for (int i = 0; i < predicts.size(); i++)
    {

        cv::Mat predict = predicts[i];

        std::vector<cv::Rect> boxes;
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<std::vector<cv::Point>> points;

        for (int i = 0; i < predict.rows; i++)
        {
            float conf = predict.at<float>(i, 4);
            if (conf < this->objConf)
            {
                continue;
            }
            cv::Mat classScores = predict.row(i).colRange(5 + 2 * this->pointNum, 5 + 2 * this->pointNum + this->classNames.size());

            cv::Point classIdPoint;
            double clsConf;
            cv::minMaxLoc(classScores, 0, &clsConf, 0, &classIdPoint);

            if (conf * clsConf < this->objConf)
            {
                continue;
            }

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

            std::vector<cv::Point> localPoints;
            localPoints.reserve(this->pointNum);

            for (int point_id = 0; point_id < this->pointNum; point_id++)
            {
                float pointX = predict.at<float>(i, 5 + 2 * point_id);
                float pointY = predict.at<float>(i, 5 + 2 * point_id + 1);
                cv::Point point = cv::Point(pointX, pointY);
                localPoints.push_back(point);
            }
            points.push_back(localPoints);
        }

        std::vector<int> indexes;
        cv::dnn::NMSBoxesBatched(boxes, confidences, classIds, this->objConf, this->nmsConf, indexes);
        std::vector<cv::Rect> outputRect;
        std::vector<float> outputConfidence;
        std::vector<string> outputName;
        std::vector<std::vector<cv::Point>> outputPoint;

        outputRect.reserve(indexes.size());
        outputConfidence.reserve(indexes.size());
        outputName.reserve(indexes.size());
        outputPoints.reserve(indexes.size());

        for (int index : indexes)
        {
            outputRect.push_back(boxes.at(index));
            outputConfidence.push_back(confidences.at(index));
            outputName.push_back(this->classNames[classIds.at(index)]);
            outputPoint.push_back(points.at(index));
        }
        faceTransformers[i].reverse(outputRect, outputPoint);

        outputRects.push_back(outputRect);
        outputConfidences.push_back(outputConfidence);
        outputNames.push_back(outputName);
        outputPoints.push_back(outputPoint);
    }
}
