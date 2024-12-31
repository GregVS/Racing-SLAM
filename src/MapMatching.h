#pragma once

#include <opencv2/opencv.hpp>
#include "Frame.h"
#include "Map.h"

namespace slam
{

// Triangulates 3D points from matches between two frames
void triangulate_points(Map &map, Frame &prevFrame, Frame &frame, const std::vector<cv::DMatch> &matches);

// Matches 3D map points to 2D keypoints
void match_map_points(Map &map, Frame &frame);

// If keypoints in prev frame are matched to a 3D map point, also use that match in the current frame
void piggyback_prev_frame_matches(Map &map, const Frame &prev_frame, Frame &frame, const std::vector<cv::DMatch> &matches);

};
