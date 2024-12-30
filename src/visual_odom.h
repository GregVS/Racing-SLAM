#pragma once

#include <opencv2/opencv.hpp>
#include "Frame.h"
#include "Map.h"
#include "Camera.h"

// Extracts ORB features from a frame
Frame extractFeatures(const cv::Mat &image, int id);

// Matches features between two consecutive frames
std::vector<cv::DMatch> matchFeatures(const Frame &prevFrame, const Frame &frame);

// Estimates camera pose in homogenous 4D coords
cv::Mat estimatePose(const Frame &prevFrame, const Frame &frame, const std::vector<cv::DMatch> &matches, const cv::Mat &K);

// Triangulates 3D points from matches between two frames
void triangulatePoints(Map& map, Frame &prevFrame, Frame &frame, const std::vector<cv::DMatch> &matches);

// Matches 3D map points to 2D keypoints
void matchMapPoints(Map &map, Frame &frame);
