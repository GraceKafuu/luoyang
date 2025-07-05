#pragma once
#include "FaceDetect.h"
#include "PoseTransformer.h"

class PoseDetect : public FaceDetect
{
protected:
    float pointConf;

public:
    PoseDetect(/* args */);
    PoseDetect(string dir);

    virtual void predict(vector<cv::Mat> images,
                         std::vector<std::vector<cv::Rect>> &outputRects,
                         std::vector<std::vector<string>> &outputNames,
                         std::vector<std::vector<float>> &outputConfidences,
                         std::vector<std::vector<std::vector<cv::Point>>> &outputPoints,
                         std::vector<std::vector<std::vector<float>>> &outputPointConfidences);

    float getPointConf();
};
