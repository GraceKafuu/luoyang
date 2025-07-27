#pragma once
#include "Include.h"
#include "Model.h"
#include "Transformer.h"

class Detect
{
private:
public:
    Detect(/* args */);

    Detect(string dir);

    ~Detect();

    int getBatchSize();

    int getClassNum();

    float getNmsConf();

    float getObjConf();

    virtual void predict(vector<cv::Mat> images,
                         vector<vector<cv::Rect>> &outputRects,
                         vector<vector<string>> &outputNames,
                         vector<vector<float>> &outputConfidences,
                         vector<vector<vector<cv::Point>>> &outputPoints,
                         vector<vector<vector<float>>> &outputPointConfidences);

    virtual void warmup();

protected:
    int batchSize;

    vector<string> classNames;

    Model *model;

    float nmsConf;

    float objConf;

    int deviceId;

    bool useNms;

};
