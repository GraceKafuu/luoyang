#include "Transformer.h"


/**
 * @brief 默认构造函数
 */
Transformer::Transformer(/* args */)
{
}

/**
 * @brief 构造函数，从文件路径加载图像
 * 
 * @param imagePath 图像文件路径
 * @param normalizeHeight 目标归一化高度
 * @param normalizeWidth 目标归一化宽度
 */
Transformer::Transformer(string imagePath, int normalizeHeight, int normalizeWidth)
{
    this->oriImage = cv::imread(imagePath);
    this->normalizeHeight = normalizeHeight;
    this->normalizeWidth = normalizeWidth;
}

/**
 * @brief 构造函数，直接使用图像矩阵
 * 
 * @param image 输入图像
 * @param normalizeHeight 目标归一化高度
 * @param normalizeWidth 目标归一化宽度
 */
Transformer::Transformer(cv::Mat image, int normalizeHeight, int normalizeWidth)
{
    this->oriImage = image;
    this->normalizeHeight = normalizeHeight;
    this->normalizeWidth = normalizeWidth;
}

/**
 * @brief 图像预处理
 * 
 * 对图像进行缩放、填充、归一化等操作，使其适合作为模型输入。
 * 主要步骤包括：
 * 1. 计算缩放比例
 * 2. 按比例缩放图像
 * 3. 在缩放后的图像周围填充边框使其达到目标尺寸
 * 4. 将图像转换为模型输入格式
 */
void Transformer::process()
{
    // 获取原始图像尺寸
    float imgHeight = static_cast<float>(this->oriImage.size().height);
    float imgWidth = static_cast<float>(this->oriImage.size().width);

    // 计算缩放比例，保持宽高比不变
    this->resizeRatio = std::min(static_cast<float>(this->normalizeHeight) / imgHeight,
                                 static_cast<float>(this->normalizeWidth) / imgWidth);

    // 计算缩放后的尺寸
    int resize_w = int(this->resizeRatio * imgWidth);
    int resize_h = int(this->resizeRatio * imgHeight);

    // 计算需要填充的边框大小
    float dw = (float)(this->normalizeWidth - resize_w) / 2.0f;
    float dh = (float)(this->normalizeHeight - resize_h) / 2.0f;

    // 缩放图像
    resize(this->oriImage, this->normalizeImage, cv::Size(resize_w, resize_h));

    // 计算填充边框的像素数
    this->top = int(std::round(dh - 0.1f));
    this->bottom = int(std::round(dh + 0.1f));
    this->left = int(std::round(dw - 0.1f));
    this->right = int(std::round(dw + 0.1f));

    // 在图像周围填充边框（使用灰色填充）
    cv::copyMakeBorder(this->normalizeImage, this->normalizeImage, top, bottom, left, right, cv::BORDER_CONSTANT, 128);

    // 将图像转换为模型输入格式（归一化、调整通道顺序等）
    cv::dnn::blobFromImage(this->normalizeImage,
                           this->inputMat,
                           1 / 255.0,
                           cv::Size(this->normalizeWidth, this->normalizeHeight),
                           cv::Scalar(0, 0, 0),
                           true,
                           false,
                           CV_32F);
}

/**
 * @brief 坐标反变换
 * 
 * 将模型输出的坐标变换回原始图像坐标系。
 * 主要用于将检测框坐标从处理后的图像坐标系转换回原始图像坐标系。
 * 注意：此方法只处理检测框坐标，不处理关键点坐标。
 * 
 * @param boxes 检测框坐标（输入输出参数）
 * @param points 关键点坐标（此基类方法中未处理）
 */
void Transformer::reverse(std::vector<cv::Rect> &boxes,
                          std::vector<std::vector<cv::Point>> &points)
{
    float imgHeight = (float)this->oriImage.size().height;
    float imgWidth = (float)this->oriImage.size().width;

    // 对每个检测框进行坐标反变换
    for (cv::Rect &box : boxes)
    {
        // 考虑填充和缩放，将坐标转换回原始图像坐标系
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

/**
 * @brief 获取归一化图像
 * 
 * @return cv::Mat 归一化后的图像
 */
cv::Mat Transformer::getNormalizeImage()
{
    return this->normalizeImage;
}

/**
 * @brief 获取模型输入图像矩阵
 * 
 * @return cv::Mat 模型输入图像矩阵
 */
cv::Mat Transformer::getInputMat()
{
    return this->inputMat;
}