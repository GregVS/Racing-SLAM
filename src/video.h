#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include "Frame.h"
#include "MapPoint.h"

cv::VideoCapture initializeVideo(const std::string &videoPath);

cv::Mat nextFrame(cv::VideoCapture &cap);

void drawMatches(const Frame &prevFrame, const Frame &frame, const std::vector<cv::DMatch> &matches);

void showCorrespondences(const MapPoint &point);