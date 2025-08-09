#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <regex>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <onnxruntime_cxx_api.h>


using namespace std;

/**
 * @brief 读取文件中的所有行
 * 
 * @param fileName 文件路径
 * @return vector<string> 文件中每行内容组成的向量
 */
vector<string> readLines(string &fileName);

/**
 * @brief 字符串分割函数
 * 
 * @param str 待分割的字符串
 * @param delim 分割符（正则表达式）
 * @return vector<string> 分割后的字符串向量
 */
vector<string> stringSplit(string &str, string delim);

/**
 * @brief 读取键值对配置文件
 * 
 * 读取格式为"key=value"的配置文件，将其转换为键值对映射
 * 
 * @param fileName 配置文件路径
 * @return unordered_map<string, string> 键值对映射
 */
unordered_map<string, string> readMap(string &fileName);

const int SKELETON_POINT_NUM = 19;                                      // 骨骼关键点数量

const int SKELETON_FIRST[SKELETON_POINT_NUM] = {15, 13, 16, 14, 11, 5, 6, 5, 5, 6, 7, 8, 1, 0, 0, 1, 2, 3, 4};  // 骨骼连接线起始点索引

const int SKELETON_SECOND[SKELETON_POINT_NUM] = {13, 11, 14, 12, 12, 11, 12, 6, 7, 8, 9, 10, 2, 1, 2, 3, 4, 5, 6}; // 骨骼连接线终点索引