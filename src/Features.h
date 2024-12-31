#pragma once

#include <opencv2/opencv.hpp>
#include "Camera.h"
#include "Frame.h"

namespace slam
{

struct PoseEstimate {
    cv::Mat pose;
    std::vector<cv::DMatch> filteredMatches;
};

Frame extract_features(const cv::Mat &image, int id);

std::vector<cv::DMatch> match_features(const Frame &prevFrame, const Frame &frame);

PoseEstimate estimate_pose(const Frame &prevFrame, const Frame &frame, const Camera &camera,
                  const std::vector<cv::DMatch> &matches);

};