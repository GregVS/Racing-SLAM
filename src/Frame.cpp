#include "Frame.h"

namespace slam {

Frame::Frame(int id, const cv::Mat &image, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors)
        : m_id(id)
        , m_image(image)
        , m_keypoints(keypoints)
        , m_descriptors(descriptors)
{
    m_pose = cv::Mat::eye(4, 4, CV_64F);
    m_map_points.resize(keypoints.size(), nullptr);

    std::vector<cv::Point2f> keypoints2f;
    for (const auto &keypoint : keypoints) {
        keypoints2f.push_back(keypoint.pt);
    }
    m_kd_tree.build(keypoints2f);
}

MapPoint *Frame::get_corresponding_map_point(int keypoint_index) const
{
    return m_map_points[keypoint_index];
}

void Frame::set_corresponding_map_point(int keypoint_index, MapPoint *map_point)
{
    m_map_points[keypoint_index] = map_point;
}

cv::Mat Frame::get_descriptor(int keypoint_index) const
{
    return m_descriptors.row(keypoint_index);
}

const cv::Mat &Frame::get_descriptors() const
{
    return m_descriptors;
}

const cv::Mat &Frame::get_image() const
{
    return m_image;
}

const std::vector<cv::KeyPoint> &Frame::get_keypoints() const
{
    return m_keypoints;
}

const cv::KeyPoint &Frame::get_keypoint(int keypoint_index) const
{
    return m_keypoints[keypoint_index];
}

const cv::Mat &Frame::get_pose() const
{
    return m_pose;
}

std::vector<size_t> Frame::get_keypoints_within_radius(const cv::Point2f &target, float radius) const
{
    return m_kd_tree.radius_search(target, radius);
}

void Frame::set_pose(const cv::Mat &pose)
{
    m_pose = pose;
}

int Frame::get_id() const
{
    return m_id;
}

};