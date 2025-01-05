#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include "Frame.h"
#include "MapPoint.h"

namespace slam
{

cv::VideoCapture init_video(const std::string &videoPath);

cv::Mat next_frame(cv::VideoCapture &cap);

void draw_matches(const Frame &prevFrame, const Frame &frame, const std::vector<cv::DMatch> &matches);

void draw_correspondences(const MapPoint &point);

void draw_features(const Frame &frame);

};