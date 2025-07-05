#pragma once
#include <regex>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <onnxruntime_cxx_api.h>


using namespace std;

vector<string> readLines(string &fileName);

vector<string> stringSplit(string &str, string delim);

unordered_map<string, string> readMap(string &fileName);

const int SKELETON_POINT_NUM = 19;

const int SKELETON_FIRST[SKELETON_POINT_NUM] = {15, 13, 16, 14, 11, 5, 6, 5, 5, 6, 7, 8, 1, 0, 0, 1, 2, 3, 4};

const int SKELETON_SECOND[SKELETON_POINT_NUM] = {13, 11, 14, 12, 12, 11, 12, 6, 7, 8, 9, 10, 2, 1, 2, 3, 4, 5, 6};