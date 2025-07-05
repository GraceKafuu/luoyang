#include "FaceTransformer.h"



FaceTransformer::FaceTransformer()
{
}

FaceTransformer::FaceTransformer(string imagePath,
                                        int normalizeHeight,
                                        int normalizeWidth,
                                        int pointNum)
{
    this->pointNum = pointNum;
    this->oriImage = cv::imread(imagePath);
    this->normalizeHeight = normalizeHeight;
    this->normalizeWidth = normalizeWidth;
}

FaceTransformer::FaceTransformer(cv::Mat image,
                                        int normalizeHeight,
                                        int normalizeWidth,
                                        int pointNum)
{
    this->pointNum = pointNum;
    this->oriImage = image;
    this->normalizeHeight = normalizeHeight;
    this->normalizeWidth = normalizeWidth;
}

void FaceTransformer::reverse(std::vector<cv::Rect> &boxes,
                              std::vector<std::vector<cv::Point>> &points)
{
    float imgHeight = (float)this->oriImage.size().height;
    float imgWidth = (float)this->oriImage.size().width;

    for (cv::Rect &box : boxes)
    {
        float left = static_cast<float>(box.x - this->left) / this->resizeRatio;
        float top = static_cast<float>(box.y - this->top) / this->resizeRatio;
        float width = box.width / this->resizeRatio;
        float height = box.height / this->resizeRatio;

        box.x = static_cast<int>(left);
        box.y = static_cast<int>(top);
        box.width = static_cast<int>(width);
        box.height = static_cast<int>(height);
    }
    for (std::vector<cv::Point> &localPoints : points)
    {
        for (cv::Point &point : localPoints)
        {
            float x = static_cast<float>(point.x - this->left) / this->resizeRatio;
            float y = static_cast<float>(point.y - this->top) / this->resizeRatio;

            point.x = static_cast<int>(x);
            point.y = static_cast<int>(y);
        }
    }
}
