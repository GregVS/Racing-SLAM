#pragma once
#include <opencv2/opencv.hpp>
#include <string>

cv::VideoCapture initializeVideo(const std::string &videoPath);

void playVideo(cv::VideoCapture &cap);

bool displayFrame(cv::VideoCapture &cap);