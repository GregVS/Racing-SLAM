#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "MapPoint.h"
#include "KDTree.h"

namespace slam
{

class Frame {
public:
    Frame(int id, const cv::Mat &image, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors);

    Frame(const Frame &other) = delete;

    Frame(Frame &&other) noexcept = default;

    MapPoint *get_corresponding_map_point(int keypoint_index) const;

    void set_corresponding_map_point(int keypoint_index, MapPoint *map_point);

    cv::Mat get_descriptor(int keypoint_index) const;

    const cv::Mat &get_descriptors() const;

    const cv::Mat &get_image() const;

    const std::vector<cv::KeyPoint> &get_keypoints() const;

    const cv::KeyPoint &get_keypoint(int keypoint_index) const;

    const cv::Mat &get_pose() const;

    std::vector<size_t> get_keypoints_within_radius(const cv::Point2f &target, float radius) const;

    void set_pose(const cv::Mat &pose);

    int get_id() const;

private:
    const int m_id;
    const cv::Mat m_image;

    const std::vector<cv::KeyPoint> m_keypoints;
    cv::Mat m_descriptors;
    KDTree2D m_kd_tree;

    cv::Mat m_pose;
    std::vector<MapPoint *> m_map_points;
};

};