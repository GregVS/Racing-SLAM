#pragma once

#include <opencv2/opencv.hpp>

struct FrameData {
    cv::Mat frame;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Mat pose;
};

// Extracts ORB features from a frame
FrameData extractFeatures(const cv::Mat &frame);

// Matches features between two consecutive frames
std::vector<cv::DMatch> matchFeatures(const FrameData &prevFrame, const FrameData &frame);

// Estimates camera pose in homogenous 4D coords
cv::Mat estimatePose(const FrameData &prevFrame, const FrameData &frame, const std::vector<cv::DMatch> &matches);

// Triangulates 3D points from matches between two frames
std::vector<cv::Point3f> triangulatePoints(const FrameData &prevFrame, const FrameData &frame, const std::vector<cv::DMatch> &matches);
