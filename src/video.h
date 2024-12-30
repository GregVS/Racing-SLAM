#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include "Frame.h"

cv::VideoCapture initializeVideo(const std::string &videoPath);

bool nextFrame(cv::VideoCapture &cap, cv::Mat &frame);

void drawMatches(const Frame &prevFrame, const Frame &frame, const std::vector<cv::DMatch> &matches);
