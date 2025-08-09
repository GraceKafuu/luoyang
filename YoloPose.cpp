#include <iostream>
#include <string>
#include <queue>
#include <thread>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "PoseDetect.h"

using namespace std;
using namespace cv;

/**
 * @brief 打印使用说明
 * 
 * 显示姿态检测程序的命令行参数使用方法和各参数的含义
 */
void print_usage() {
    cout << "Usage: ./luoyang_yolo_pose <model_dir> <image_path> [output_path]" << endl;
    cout << "Arguments:" << endl;
    cout << "  model_dir    : Path to YOLO model directory" << endl;
    cout << "  image_path   : Path to input image" << endl;
    cout << "  output_path  : (Optional) Path to save output image" << endl;
}


/**
 * @brief 执行姿态检测
 * 
 * 加载姿态检测模型，对指定图像进行人体姿态检测，并绘制骨骼连接线显示姿态结果
 * 
 * @param model_dir 模型目录路径
 * @param image_path 输入图像路径
 * @param output_path 输出图像路径（可选）
 */
void object_pose_detection(const string& model_dir, const string& image_path, const string& output_path = "") {
    try {
        // 创建姿态检测器实例
        PoseDetect detect(model_dir);
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

        // 执行姿态检测
        detect.predict(images,
                      outputRects,
                      outputNames,
                      outputConfidences,
                      points,
                      pointConfidences);
        
        // 在图像上绘制检测结果和姿态骨骼
        for (int i = 0; i < outputRects[0].size(); i++) {
            cv::Rect box = outputRects[0].at(i);
            
            // 绘制人体边界框
            cv::rectangle(image, box, cv::Scalar(0, 0, 255), 2, 8);
            
            // 添加标签文本
            putText(image, outputNames[0].at(i), cv::Point(box.x, box.y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
            
            // 获取当前人体的关键点
            std::vector<cv::Point> localPoints = points[0][i];
            
            // 根据骨骼连接关系绘制连接线
            for (int point_index = 0; point_index < SKELETON_POINT_NUM; point_index++){
                // 检查关键点置信度是否满足阈值要求
                if (pointConfidences[0][i][SKELETON_FIRST[point_index]] >= detect.getPointConf() and
                 pointConfidences[0][i][SKELETON_FIRST[point_index]] >= detect.getPointConf()){
                    // 绘制骨骼连接线
                    cv::line(image, localPoints[SKELETON_FIRST[point_index]], localPoints[SKELETON_SECOND[point_index]], cv::Scalar(255, 0, 255));
                }
            }
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
 * 程序入口点，解析命令行参数并调用姿态检测函数
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

    // 执行姿态检测
    object_pose_detection(model_dir, image_path, output_path);
    return 0;
}