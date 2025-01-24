#include "Frame.h"

namespace slam {

// Frame
Frame::Frame(int index, const cv::Mat& image, const ExtractedFeatures& features)
    : m_index(index), m_image(image), m_features(features), m_pose(Eigen::Matrix4f::Identity())
{
    std::vector<Eigen::Vector2f> keypoints;
    for (const auto& keypoint : m_features.keypoints) {
        keypoints.push_back(Eigen::Vector2f(keypoint.pt.x, keypoint.pt.y));
    }
    m_kd_tree.build(keypoints);
    m_map_matches.resize(m_features.keypoints.size(), nullptr);
}

size_t Frame::index() const
{
    return m_index;
}

const cv::Mat& Frame::image() const
{
    return m_image;
}

const ExtractedFeatures& Frame::features() const
{
    return m_features;
}

const Eigen::Matrix4f& Frame::pose() const
{
    return m_pose;
}

void Frame::set_pose(const Eigen::Matrix4f& pose)
{
    m_pose = pose;
}

void Frame::add_map_match(const MapPointMatch& match)
{
    m_map_matches[match.keypoint_index] = &match.point;
    m_matched_map_points.insert(&match.point);
}

void Frame::remove_map_match(const MapPointMatch& match)
{
    m_map_matches[match.keypoint_index] = nullptr;
    m_matched_map_points.erase(&match.point);
}

const MapPoint& Frame::map_match(size_t index) const
{
    return *m_map_matches[index];
}

size_t Frame::num_map_matches() const
{
    return std::count_if(m_map_matches.begin(), m_map_matches.end(), [](const MapPoint* point) {
        return point != nullptr;
    });
}

const cv::Mat Frame::descriptor(size_t index) const
{
    return m_features.descriptors.row(index);
}

const cv::KeyPoint& Frame::keypoint(size_t index) const
{
    return m_features.keypoints[index];
}

std::vector<size_t> Frame::features_in_region(const Eigen::Vector2f& uv, float radius) const
{
    return m_kd_tree.radius_search(uv, radius);
}

bool Frame::is_matched(size_t keypoint_index) const
{
    return m_map_matches[keypoint_index] != nullptr;
}

bool Frame::is_matched(const MapPoint& point) const
{
    return m_matched_map_points.find(&point) != m_matched_map_points.end();
}

// MapPointIterator
Frame::MapPointIterator::MapPointIterator(const Frame& frame, size_t index)
    : m_frame(frame), m_index(index)
{
    while (m_index < m_frame.m_map_matches.size() && m_frame.m_map_matches[m_index] == nullptr) {
        m_index++;
    }
}

MapPointMatch Frame::MapPointIterator::operator*() const
{
    return MapPointMatch{*m_frame.m_map_matches[m_index], m_index};
}

Frame::MapPointIterator& Frame::MapPointIterator::operator++()
{
    m_index++;
    while (m_index < m_frame.m_map_matches.size() && m_frame.m_map_matches[m_index] == nullptr) {
        m_index++;
    }
    return *this;
}

bool Frame::MapPointIterator::operator!=(const MapPointIterator& other) const
{
    return m_index != other.m_index;
}

Frame::MapPointIterator Frame::map_matches() const
{
    return MapPointIterator(*this, 0);
}

Frame::MapPointIterator Frame::MapPointIterator::begin() const
{
    return MapPointIterator(m_frame, 0);
}

Frame::MapPointIterator Frame::MapPointIterator::end() const
{
    return MapPointIterator(m_frame, m_frame.m_map_matches.size());
}

} // namespace slam