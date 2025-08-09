#include <iostream>
#include <string>
#include <queue>
#include <thread>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "FaceDetect.h"

using namespace std;
using namespace cv;

/**
 * @brief 打印使用说明
 * 
 * 显示人脸检测程序的命令行参数使用方法和各参数的含义
 */
void print_usage() {
    cout << "Usage: ./luoyang_yolo_face <model_dir> <image_path> [output_path]" << endl;
    cout << "Arguments:" << endl;
    cout << "  model_dir    : Path to YOLO model directory" << endl;
    cout << "  image_path   : Path to input image" << endl;
    cout << "  output_path  : (Optional) Path to save output image" << endl;
}


/**
 * @brief 执行人脸检测
 * 
 * 加载人脸检测模型，对指定图像进行人脸检测和关键点识别，并显示或保存结果
 * 
 * @param model_dir 模型目录路径
 * @param image_path 输入图像路径
 * @param output_path 输出图像路径（可选）
 */
void object_face_detection(const string& model_dir, const string& image_path, const string& output_path = "") {
    try {
        // 创建人脸检测器实例
        FaceDetect detect(model_dir);
        cv::Mat image = cv::imread(image_path);
        
        // 检查图像是否成功加载
        if (image.empty()) {
            cerr << "Error: Could not read image from " << image_path << endl;
            return;
        }

        // 准备输出容器
        std::vector<std::vector<cv::Rect>> outputRects;
        std::vector<std::vector<string>> outputNames;
        std::vector<std::vector<float>> outputConfidences;
        std::vector<std::vector<std::vector<cv::Point>>> points;
        std::vector<std::vector<std::vector<float>>> pointConfidences;
        vector<cv::Mat> images;
        images.push_back(image);

        // 执行人脸检测
        detect.predict(images,
                      outputRects,
                      outputNames,
                      outputConfidences,
                      points,
                      pointConfidences);
        
        // 在图像上绘制检测结果和关键点
        for (int i = 0; i < outputRects[0].size(); i++){
            cv::Rect box = outputRects[0][i];
            std::vector<cv::Point> localPoints = points[0][i];
            
            // 绘制人脸关键点
            for (cv::Point point : localPoints){
                cv::circle(image, point, 2, cv::Scalar(0, 255, 255));
            }
            
            // 绘制人脸边界框
            cv::rectangle(image, box, cv::Scalar(0, 0, 255), 2, 8);
            
            // 添加标签文本
            putText(image, outputNames[0].at(i), cv::Point(box.x, box.y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
        }

        // 保存或显示结果
        if (!output_path.empty()) {
            cv::imwrite(output_path, image);
            cout << "Result saved to: " << output_path << endl;
        } else {
            cv::imshow("detect", image);
            cv::waitKeyEx();
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
}

/**
 * @brief 主函数
 * 
 * 程序入口点，解析命令行参数并调用人脸检测函数
 * 
 * @param argc 命令行参数数量
 * @param argv 命令行参数数组
 * @return int 程序退出码
 */
int main(int argc, char* argv[]) {
    // 检查参数数量
    if (argc < 3) {
        print_usage();
        return -1;
    }

    // 解析命令行参数
    string model_dir = argv[1];
    string image_path = argv[2];
    string output_path = (argc > 3) ? argv[3] : "";

    // 执行人脸检测
    object_face_detection(model_dir, image_path, output_path);
    return 0;
}