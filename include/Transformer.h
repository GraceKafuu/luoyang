#pragma once
#include "Include.h"

class Transformer
{
protected:
    cv::Mat oriImage;

    cv::Mat normalizeImage;

    cv::Mat inputMat;

    int normalizeHeight;

    int normalizeWidth;

    int top;

    int bottom;

    int left;

    int right;

    float resizeRatio;

public:
    Transformer(/* args */);

    Transformer(string imagePath, int normalizeHeight, int normalizeWidth);

    Transformer(cv::Mat image, int normalizeHeight, int normalizeWidth);

    void process();

    virtual void reverse(std::vector<cv::Rect> &boxes,
                         std::vector<std::vector<cv::Point>> &points);

    cv::Mat getNormalizeImage();

    cv::Mat getInputMat();
};
