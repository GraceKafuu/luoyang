#include "Transformer.h"


Transformer::Transformer(/* args */)
{
}

Transformer::Transformer(string imagePath, int normalizeHeight, int normalizeWidth)
{
    this->oriImage = cv::imread(imagePath);
    this->normalizeHeight = normalizeHeight;
    this->normalizeWidth = normalizeWidth;
}

Transformer::Transformer(cv::Mat image, int normalizeHeight, int normalizeWidth)
{
    this->oriImage = image;
    this->normalizeHeight = normalizeHeight;
    this->normalizeWidth = normalizeWidth;
}

void Transformer::process()
{
    float imgHeight = static_cast<float>(this->oriImage.size().height);
    float imgWidth = static_cast<float>(this->oriImage.size().width);

    this->resizeRatio = std::min(static_cast<float>(this->normalizeHeight) / imgHeight,
                                 static_cast<float>(this->normalizeWidth) / imgWidth);

    int resize_w = int(this->resizeRatio * imgWidth);
    int resize_h = int(this->resizeRatio * imgHeight);

    float dw = (float)(this->normalizeWidth - resize_w) / 2.0f;
    float dh = (float)(this->normalizeHeight - resize_h) / 2.0f;

    resize(this->oriImage, this->normalizeImage, cv::Size(resize_w, resize_h));

    this->top = int(std::round(dh - 0.1f));
    this->bottom = int(std::round(dh + 0.1f));
    this->left = int(std::round(dw - 0.1f));
    this->right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(this->normalizeImage, this->normalizeImage, top, bottom, left, right, cv::BORDER_CONSTANT, 128);

    cv::dnn::blobFromImage(this->normalizeImage,
                           this->inputMat,
                           1 / 255.0,
                           cv::Size(this->normalizeWidth, this->normalizeHeight),
                           cv::Scalar(0, 0, 0),
                           true,
                           false,
                           CV_32F);
}

void Transformer::reverse(std::vector<cv::Rect> &boxes,
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
}

cv::Mat Transformer::getNormalizeImage()
{
    return this->normalizeImage;
}

cv::Mat Transformer::getInputMat()
{
    return this->inputMat;
}
