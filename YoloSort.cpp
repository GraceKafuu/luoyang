#include <iostream>

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 状态变量: [x, y, vx, vy] (位置和速度)
    const int stateNum = 4;
    // 测量变量: [x, y] (只能测量位置)
    const int measureNum = 2;
    
    cv::KalmanFilter KF(stateNum, measureNum, 0);
    
    // 初始化转移矩阵 (A)
    KF.transitionMatrix = (cv::Mat_<float>(stateNum, stateNum) << 
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);
    
    // 初始化测量矩阵 (H)
    KF.measurementMatrix = (cv::Mat_<float>(measureNum, stateNum) << 
        1, 0, 0, 0,
        0, 1, 0, 0);
    
    // 初始化过程噪声协方差矩阵 (Q)
    setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-4));
    
    // 初始化测量噪声协方差矩阵 (R)
    setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));
    
    // 初始化后验误差协方差矩阵 (P)
    setIdentity(KF.errorCovPost, cv::Scalar::all(1));
    
    // 初始状态
    cv::Mat statePost = (cv::Mat_<float>(stateNum, 1) << 0, 0, 0, 0);
    KF.statePost = statePost;
    
    // 模拟测量数据 (带噪声的真实位置)
    std::vector<cv::Point> measurements;
    for (int i = 0; i < 100; i++) {
        measurements.push_back(cv::Point(i + rand() % 3, i + rand() % 3));
    }
    
    // 用于存储预测和修正结果
    std::vector<cv::Point> predictions;
    std::vector<cv::Point> corrections;
    
    // 卡尔曼滤波过程
    for (const auto& meas : measurements) {
        // 预测阶段
        cv::Mat prediction = KF.predict();
        predictions.push_back(cv::Point(prediction.at<float>(0), prediction.at<float>(1)));
        
        // 测量值
        cv::Mat measurement = (cv::Mat_<float>(measureNum, 1) << meas.x, meas.y);
        
        // 更新阶段
        cv::Mat estimated = KF.correct(measurement);
        corrections.push_back(cv::Point(estimated.at<float>(0), estimated.at<float>(1)));
    }
    
    // 可视化结果
    cv::Mat img(500, 500, CV_8UC3, cv::Scalar(0));
    for (size_t i = 1; i < measurements.size(); i++) {
        // 测量值 (红色)
        cv::line(img, measurements[i-1]*5, measurements[i]*5, cv::Scalar(0, 0, 255), 1);
        // 预测值 (绿色)
        cv::line(img, predictions[i-1]*5, predictions[i]*5, cv::Scalar(0, 255, 0), 1);
        // 修正值 (蓝色)
        cv::line(img, corrections[i-1]*5, corrections[i]*5, cv::Scalar(255, 0, 0), 1);
    }
    
    cv::imshow("Kalman Filter", img);
    cv::waitKey(0);
    
    return 0;
}