#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Detect.h"
#include "BYTETracker.h"

using namespace std;
using namespace cv;

/**
 * @brief 打印程序使用方法
 * 
 * 显示程序的正确使用方式及各参数含义
 */
void print_usage() {
    cout << "Usage: ./luoyang_yolo_track <model_dir> <video_path> [output_video_path] [target_class]" << endl;
    cout << "Arguments:" << endl;
    cout << "  model_dir         : Path to YOLO model directory" << endl;
    cout << "  video_path        : Path to input video" << endl;
    cout << "  output_video_path : (Optional) Path to save output video" << endl;
    cout << "  target_class      : (Optional) Target class to track (default: person)" << endl;
}

/**
 * @brief 执行目标跟踪
 * 
 * 使用YOLO模型进行目标检测，并结合BYTETracker进行多目标跟踪
 * 
 * @param model_dir 模型目录路径
 * @param video_path 输入视频路径
 * @param output_path 输出视频路径（可选）
 * @param target_class 目标跟踪类别（默认为person）
 */
void object_tracking(const string& model_dir, const string& video_path, const string& output_path = "", const string& target_class = "person") {
    try {
        // 创建检测器实例
        Detect detect(model_dir);
        
        // 打开视频文件
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            cerr << "Error: Could not open video from " << video_path << endl;
            return;
        }
        
        // 获取视频属性
        int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(CAP_PROP_FPS);
        
        // 初始化视频写入器（如果指定了输出路径）
        cv::VideoWriter writer;
        bool save_video = !output_path.empty();
        if (save_video) {
            writer = cv::VideoWriter(output_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, 
                                   cv::Size(frame_width, frame_height));
            if (!writer.isOpened()) {
                cerr << "Error: Could not create video writer for " << output_path << endl;
                return;
            }
        }

        // 初始化跟踪器
        BYTETracker tracker(fps, 30);
        
        cv::Mat frame;
        int frame_id = 0;
        
        cout << "Starting object tracking..." << endl;
        
        // 逐帧处理视频
        while (true) {
            cap >> frame;
            if (frame.empty()) {
                break;
            }
            
            frame_id++;
            
            // 进行目标检测
            // 定义检测结果容器
            std::vector<std::vector<cv::Rect>> outputRects;        // 存储检测到的目标边界框
            std::vector<std::vector<string>> outputNames;         // 存储检测到的目标类别名称
            std::vector<std::vector<float>> outputConfidences;    // 存储检测置信度
            std::vector<std::vector<std::vector<cv::Point>>> points;        // 存储关键点坐标
            std::vector<std::vector<std::vector<float>>> pointConfidences;  // 存储关键点置信度
            
            // 准备输入图像
            vector<cv::Mat> images;
            images.push_back(frame);
            
            // 执行检测
            detect.predict(images,
                          outputRects,
                          outputNames,
                          outputConfidences,
                          points,
                          pointConfidences);
         
            // 将检测结果转换为跟踪器所需的格式
            std::vector<detect_result> detect_results;
            for (int i = 0; i < outputRects[0].size(); i++) {
                // 只跟踪指定类别的目标
                if(outputNames[0][i] == target_class){
                    detect_result result;
                    result.classId = 0; 
                    result.confidence = outputConfidences[0][i];
                    result.box = outputRects[0][i];
                    detect_results.push_back(result);
                }
            }
      
            // 使用BYTETracker更新跟踪状态
            std::vector<STrack> tracking_results = tracker.update(detect_results);
            
            // 在图像上绘制检测框和跟踪ID
            for (int i = 0; i < tracking_results.size(); i++) {
                STrack track = tracking_results[i];
                
                // 获取跟踪框并转换为绘制格式
                track.static_tlbr(); // 确保tlbr坐标已计算
                cv::Rect box = cv::Rect(track.tlbr[0], track.tlbr[1], track.tlbr[2] - track.tlbr[0], track.tlbr[3] - track.tlbr[1]);
                
                // 绘制边界框
                // 使用track_id对应的颜色，提高多目标的可视化区分度
                cv::rectangle(frame, box, tracker.get_color(track.track_id), 2, 8);
                
                // 绘制跟踪ID
                char track_id_str[10];
                sprintf(track_id_str, "%d", track.track_id);
                
                // 在边界框顶部显示跟踪ID
                cv::putText(frame, track_id_str, cv::Point(box.x, box.y - 10), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, tracker.get_color(track.track_id), 2);
            }
            
            // 保存或显示结果帧
            if (save_video) {
                writer.write(frame);
            } else {
                cv::imshow("Object Tracking", frame);
                if (cv::waitKey(1) == 27) { // ESC键退出
                    break;
                }
            }
            
            // 输出进度
            if (frame_id % 30 == 0) {
                cout << "Processed " << frame_id << " frames..." << endl;
            }
        }
        
        // 释放资源
        cap.release();
        if (save_video) {
            writer.release();
            cout << "Result saved to: " << output_path << endl;
        }
        cv::destroyAllWindows();
        
        cout << "Tracking completed. Total frames processed: " << frame_id << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
}

/**
 * @brief 主函数
 * 
 * 程序入口点，解析命令行参数并调用目标跟踪函数
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
    string video_path = argv[2];
    string output_path = (argc > 3) ? argv[3] : "";
    string target_class = (argc > 4) ? argv[4] : "person";

    // 开始执行目标跟踪任务
    object_tracking(model_dir, video_path, output_path, target_class);
    return 0;
}