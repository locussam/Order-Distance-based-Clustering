#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <string>

void get_data_and_name(cv::Mat&data, std::vector<std::string>&feature_name_v,const std::string path = "./data/featureFolder/");