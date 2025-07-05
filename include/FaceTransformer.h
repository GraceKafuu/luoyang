#pragma once
#include "Transformer.h"

class FaceTransformer : public Transformer
{

protected:
    int pointNum;

public:
    FaceTransformer();

    FaceTransformer(string imagePath,
                    int normalizeHeight,
                    int normalizeWidth,
                    int pointNum);

    FaceTransformer(cv::Mat image,
                    int normalizeHeight,
                    int normalizeWidth,
                    int pointNum);

    virtual void reverse(std::vector<cv::Rect> &boxes, std::vector<std::vector<cv::Point>> &points);
};
