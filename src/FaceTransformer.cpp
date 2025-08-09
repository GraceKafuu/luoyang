#include "FaceTransformer.h"



/**
 * @brief 默认构造函数
 */
FaceTransformer::FaceTransformer()
{
}

/**
 * @brief 构造函数，从文件路径加载图像
 * 
 * @param imagePath 图像文件路径
 * @param normalizeHeight 目标归一化高度
 * @param normalizeWidth 目标归一化宽度
 * @param pointNum 人脸关键点数量
 */
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

/**
 * @brief 构造函数，直接使用图像矩阵
 * 
 * @param image 输入图像
 * @param normalizeHeight 目标归一化高度
 * @param normalizeWidth 目标归一化宽度
 * @param pointNum 人脸关键点数量
 */
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

/**
 * @brief 坐标反变换
 * 
 * 重写父类的reverse方法，除了对检测框进行坐标反变换外，
 * 还对人脸关键点坐标进行反变换。将模型输出的坐标变换回原始图像坐标系。
 * 
 * @param boxes 检测框坐标（输入输出参数）
 * @param points 人脸关键点坐标（输入输出参数）
 */
void FaceTransformer::reverse(std::vector<cv::Rect> &boxes,
                              std::vector<std::vector<cv::Point>> &points)
{
    float imgHeight = (float)this->oriImage.size().height;
    float imgWidth = (float)this->oriImage.size().width;

    // 对检测框进行坐标反变换
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
    
    // 对人脸关键点进行坐标反变换
    for (std::vector<cv::Point> &localPoints : points)
    {
        for (cv::Point &point : localPoints)
        {
            // 考虑填充和缩放，将关键点坐标转换回原始图像坐标系
            float x = static_cast<float>(point.x - this->left) / this->resizeRatio;
            float y = static_cast<float>(point.y - this->top) / this->resizeRatio;

            point.x = static_cast<int>(x);
            point.y = static_cast<int>(y);
        }
    }
}