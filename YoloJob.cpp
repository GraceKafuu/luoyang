#include <iostream>
#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <future>
#include <csignal>
#include <atomic>
#include "include/Detect.h"

struct Job
{
    cv::Mat inputImage;
    std::shared_ptr<std::promise<cv::Mat>> outputImage;
};

// 共享队列
std::queue<Job> jobs;
// 互斥锁
std::mutex mutex_lock;
// 条件变量
std::condition_variable condition_v;
// 结束标记符
std::atomic<bool> finished(false);
// 中断标记符
std::atomic<bool> interrupted(false);
// 加载延迟时间
const int load_delay = 10;

// 信号处理函数
void signal_handler(int signal)
{
    std::cout << "接收到信号 " << signal << ", 准备退出..." << std::endl;
    interrupted.store(true);
    {
        std::unique_lock<std::mutex> lock(mutex_lock);
        finished.store(true);
    }
    // 通知所有 生产者和消费者 退出
    condition_v.notify_all();
}

void display(std::vector<std::future<cv::Mat>> &futures,
             int &frameCount,
             std::chrono::_V2::system_clock::time_point &start,
             cv::VideoWriter &writer)
{
    for (auto &future : futures)
    {
        if (future.valid())
        {
            cv::Mat result = future.get();
            frameCount++;
            std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
            double fps = frameCount / elapsed.count();

            std::string fpsString = "FPS: " + std::to_string((int)fps);
            cv::putText(result, fpsString, cv::Point(result.cols - 150, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            // 写入文件
            if (writer.isOpened()){
                writer.write(result);
            }
            cv::imshow("yolo", result);
            cv::waitKey(1);
        }
    }
}

void capture(const string& video_path, const string& save_path, const int& consumer_n, const int& limit)
{
    // 设置信号处理
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = signal_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
    sigaction(SIGTSTP, &sigIntHandler, NULL);

    auto start = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    cv::VideoCapture cap(video_path);

    // 获取视频的帧率、宽度和高度
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // 使用 H.264 编码器
    int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1'); // H.264 编码器

    cv::VideoWriter writer;
    if (!save_path.empty()) { 
        writer.open(save_path, fourcc, fps, cv::Size(width, height));
    }

    cv::Mat frame;
    std::vector<std::future<cv::Mat>> futures;
    while (cap.read(frame) && !interrupted.load())
    {
        Job job;
        {
            std::unique_lock<std::mutex> lock(mutex_lock);
            // 等待队列未满或生产完成
            condition_v.wait(lock, [&]() -> bool
                             { return jobs.size() < limit || finished.load() || interrupted.load(); });
            
            if (finished.load() || interrupted.load()) {
                break;
            }

            // 深拷贝
            job.inputImage = frame.clone();
            job.outputImage.reset(new std::promise<cv::Mat>());
            jobs.push(job);
            futures.push_back(job.outputImage->get_future());
        }
        // 通知消费者
        condition_v.notify_all();
        // 生产者最大等待数目
        if (futures.size() >= consumer_n + 2)
        {
            // 显示
            display(futures, frameCount, start, writer);
            futures.clear();
        }
    }
    // 最后几张图片显示
    for (auto &future : futures)
    {
        display(futures, frameCount, start, writer);
    }
    futures.clear();
    {
        // 标记已经完成
        std::cout << "生产者退出 " << frameCount << std::endl;
        std::unique_lock<std::mutex> lock(mutex_lock);
        finished.store(true);
    }
    // 释放
    cap.release();
    if (writer.isOpened()){
        writer.release();
    }
    cv::destroyAllWindows();
    // 通知消费者
    condition_v.notify_all();
}

void infer(const int& id, const string& model_dir)
{
    // 避免同时启动 (瞬时的gpu显存占用过多)
    std::this_thread::sleep_for(std::chrono::seconds(id * load_delay));
    Detect detect(model_dir);
    // 预热
    detect.warmup();
    while (!interrupted.load())
    {
        std::unique_lock<std::mutex> lock(mutex_lock);
        // 等待队列非空或者生产完成
        condition_v.wait(lock, [&]() -> bool
                         { return !jobs.empty() || finished.load() || interrupted.load(); });
        // 如果生产完成且队列为空，则退出
        if ((finished.load() && jobs.empty()) || interrupted.load())
        {
            std::cout << "消费者退出 " << id << std::endl;
            break;
        }

        if (!jobs.empty())
        {
            // 预测
            std::vector<std::vector<cv::Rect>> outputRects;
            std::vector<std::vector<string>> outputNames;
            std::vector<std::vector<float>> outputConfidences;
            std::vector<std::vector<std::vector<cv::Point>>> points;
            std::vector<std::vector<std::vector<float>>> pointConfidences;
            vector<cv::Mat> images;
            auto job = jobs.front();
            jobs.pop();
            cv::Mat image = job.inputImage;
            images.push_back(image);
            // 提前释放锁
            lock.unlock();
            detect.predict(images,
                           outputRects,
                           outputNames,
                           outputConfidences,
                           points,
                           pointConfidences);

            auto outputRect = outputRects.at(0);
            auto outputConfidence = outputConfidences.at(0);

            for (int i = 0; i < outputRect.size(); i++)
            {
                auto box = outputRect.at(i);
                cv::rectangle(image, box, cv::Scalar(0, 0, 255), 2, 8);
                putText(image, outputNames[0].at(i) + std::to_string(outputConfidences[0].at(i)), cv::Point(box.x + 10, box.y + 10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            }
            // 将结果返回给生产者
            job.outputImage->set_value(image);
            // 通知生产者
            condition_v.notify_all();
        }
    }
}

void print_usage() {
    cout << "Usage: ./luoyang_producer_consumer <model_dir> <consumer_num> <video_path> [output_path]" << endl;
    cout << "Arguments:" << endl;
    cout << "  model_dir    : Path to YOLO model directory" << endl;
    cout << "  consumer_num : Number of consumer threads" << endl;
    cout << "  video_path   : Path to input video" << endl;
    cout << "  output_path  : (Optional) Path to save output video" << endl;
}

int main(int argc, char* argv[])
{
    if (argc < 4) {
        print_usage();
        return -1;
    }

    string model_dir = argv[1];
    int consumer_n = std::stoi(argv[2]);
    string video_path = argv[3];
    string output_path = (argc > 4) ? argv[4] : "";
    // 队列最大容量    
    int limit = 5 * consumer_n;
    
    std::vector<std::thread> consumers;

    for (int i = 0; i < consumer_n; i++)
    {
        consumers.emplace_back(infer, i, model_dir);
    }
    // 等所有消费者启动好 再开始启动生产者
    std::this_thread::sleep_for(std::chrono::seconds(consumer_n == 1 ? 0 : consumer_n * load_delay));

    std::thread capture_thread(capture, video_path, output_path, consumer_n, limit);

    // 等待生产者线程结束
    capture_thread.join();

    for (auto &consumer : consumers)
    {
        // 等待消费者结束
        consumer.join();
    }

    std::cout << "程序退出" << std::endl;
    return 0;
}