#include "PoseTransformer.h"


PoseTransformer::PoseTransformer()
{
}

PoseTransformer::PoseTransformer(cv::Mat image,
                                 int normalizeHeight,
                                 int normalizeWidth,
                                 int pointNum)
{
    this->pointNum = pointNum;
    this->oriImage = image;
    this->normalizeHeight = normalizeHeight;
    this->normalizeWidth = normalizeWidth;
}

PoseTransformer::PoseTransformer(string imagePath,
                                 int normalizeHeight,
                                 int normalizeWidth,
                                 int pointNum)
{
    this->pointNum = pointNum;
    this->oriImage = cv::imread(imagePath);
    this->normalizeHeight = normalizeHeight;
    this->normalizeWidth = normalizeWidth;
}

