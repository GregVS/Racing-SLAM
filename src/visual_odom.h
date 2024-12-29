#pragma once

#include <opencv2/opencv.hpp>
#include "map.h"

// Extracts ORB features from a frame
Frame extractFeatures(const cv::Mat &image, int id);

// Matches features between two consecutive frames
std::vector<cv::DMatch> matchFeatures(const Frame &prevFrame, const Frame &frame);

// Estimates camera pose in homogenous 4D coords
cv::Mat estimatePose(const Frame &prevFrame, const Frame &frame, const std::vector<cv::DMatch> &matches);

// Triangulates 3D points from matches between two frames
void triangulatePoints(Map& map, const Frame &prevFrame, Frame &frame, const std::vector<cv::DMatch> &matches);

// Matches 3D map points to 2D keypoints
void matchMapPoints(Map &map, Frame &frame, const Camera &camera);
