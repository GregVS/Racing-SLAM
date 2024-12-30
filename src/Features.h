#pragma once

#include <opencv2/opencv.hpp>
#include "Camera.h"
#include "Frame.h"

struct PoseEstimate {
    cv::Mat pose;
    std::vector<cv::DMatch> filteredMatches;
};

Frame extractFeatures(const cv::Mat &image, int id);

std::vector<cv::DMatch> matchFeatures(const Frame &prevFrame, const Frame &frame);

PoseEstimate estimatePose(const Frame &prevFrame, const Frame &frame, const Camera &camera,
                  const std::vector<cv::DMatch> &matches);
