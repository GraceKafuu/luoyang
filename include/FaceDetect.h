#pragma once
#include "Detect.h"
#include "FaceTransformer.h"

class FaceDetect : public Detect
{
protected:
    int pointNum;

public:
    FaceDetect();
    FaceDetect(string dir);

    virtual void predict(vector<cv::Mat> images,
                         std::vector<std::vector<cv::Rect>> &outputRects,
                         std::vector<std::vector<string>> &outputNames,
                         std::vector<std::vector<float>> &outputConfidences,
                         std::vector<std::vector<std::vector<cv::Point>>> &outputPoints,
                         std::vector<std::vector<std::vector<float>>> &outputPointConfidences);
};
