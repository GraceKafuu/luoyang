#include <iostream>
#include <string>
#include <queue>
#include <thread>
#include <chrono>
#include "Detect.h"
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/**
 * @brief 打印使用说明
 * 
 * 显示程序的命令行参数使用方法和各参数的含义
 */
void print_usage() {
    cout << "Usage: ./luoyang_yolo <model_dir> <image_path> [output_path]" << endl;
    cout << "Arguments:" << endl;
    cout << "  model_dir    : Path to YOLO model directory" << endl;
    cout << "  image_path   : Path to input image" << endl;
    cout << "  output_path  : (Optional) Path to save output image" << endl;
}

/**
 * @brief 执行目标检测
 * 
 * 加载YOLO模型，对指定图像进行目标检测，并显示或保存结果
 * 
 * @param model_dir 模型目录路径
 * @param image_path 输入图像路径
 * @param output_path 输出图像路径（可选）
 */
void object_detection(const string& model_dir, const string& image_path, const string& output_path = "") {
    try {
        // 创建检测器实例
        Detect detect(model_dir);
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
        
        // 执行目标检测
        detect.predict(images,
                      outputRects,
                      outputNames,
                      outputConfidences,
                      points,
                      pointConfidences);
        
        // 在图像上绘制检测结果
        for (int i = 0; i < outputRects[0].size(); i++) {
            cv::Rect box = outputRects[0].at(i);
            cv::rectangle(image, box, cv::Scalar(0, 0, 255), 2, 8);
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
        cerr << "Error: ....................." << e.what() << endl;
    }
}

/**
 * @brief 主函数
 * 
 * 程序入口点，解析命令行参数并调用目标检测函数
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

    // 执行目标检测
    object_detection(model_dir, image_path, output_path);
    return 0;
}