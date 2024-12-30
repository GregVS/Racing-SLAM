#pragma once

#include <opencv2/opencv.hpp>
#include "Frame.h"
#include "Map.h"
#include "Camera.h"

// Triangulates 3D points from matches between two frames
void triangulatePoints(Map& map, Frame &prevFrame, Frame &frame, const std::vector<cv::DMatch> &matches);

// Matches 3D map points to 2D keypoints
void matchMapPoints(Map &map, Frame &frame);
