#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include "visual_odom.h"

cv::VideoCapture initializeVideo(const std::string &videoPath);

bool nextFrame(cv::VideoCapture &cap, cv::Mat &frame);

void drawMatches(const FrameData &prevFrame, const FrameData &frame, const std::vector<cv::DMatch> &matches);