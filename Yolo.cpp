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

void print_usage() {
    cout << "Usage: ./luoyang_yolo <model_dir> <image_path> [output_path]" << endl;
    cout << "Arguments:" << endl;
    cout << "  model_dir    : Path to YOLO model directory" << endl;
    cout << "  image_path   : Path to input image" << endl;
    cout << "  output_path  : (Optional) Path to save output image" << endl;
}

void object_detection(const string& model_dir, const string& image_path, const string& output_path = "") {
    try {
        Detect detect(model_dir);
        cv::Mat image = cv::imread(image_path);
        
        if (image.empty()) {
            cerr << "Error: Could not read image from " << image_path << endl;
            return;
        }

        std::vector<std::vector<cv::Rect>> outputRects;
        std::vector<std::vector<string>> outputNames;
        std::vector<std::vector<float>> outputConfidences;
        std::vector<std::vector<std::vector<cv::Point>>> points;
        std::vector<std::vector<std::vector<float>>> pointConfidences;
        vector<cv::Mat> images;
        images.push_back(image);
        
        detect.predict(images,
                      outputRects,
                      outputNames,
                      outputConfidences,
                      points,
                      pointConfidences);
        
        for (int i = 0; i < outputRects[0].size(); i++) {
            cv::Rect box = outputRects[0].at(i);
            cv::rectangle(image, box, cv::Scalar(0, 0, 255), 2, 8);
            putText(image, outputNames[0].at(i), cv::Point(box.x, box.y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
        }

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

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage();
        return -1;
    }

    string model_dir = argv[1];
    string image_path = argv[2];
    string output_path = (argc > 3) ? argv[3] : "";

    object_detection(model_dir, image_path, output_path);
    return 0;
}