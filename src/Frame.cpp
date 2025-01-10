#include "Frame.h"

namespace slam {

Frame::Frame(int index, const cv::Mat& image, const ExtractedFeatures& features)
    : m_index(index), m_image(image), m_features(features)
{
    std::vector<Eigen::Vector2f> keypoints;
    for (const auto& keypoint : m_features.keypoints) {
        keypoints.push_back(Eigen::Vector2f(keypoint.pt.x, keypoint.pt.y));
    }
    m_kd_tree.build(keypoints);
}

size_t Frame::index() const { return m_index; }
const cv::Mat& Frame::image() const { return m_image; }

const ExtractedFeatures& Frame::features() const { return m_features; }

void Frame::add_map_match(const MapPointMatch& match) { m_map_matches.push_back(match); }

const cv::Mat Frame::descriptor(size_t index) const { return m_features.descriptors.row(index); }

const cv::KeyPoint& Frame::keypoint(size_t index) const { return m_features.keypoints[index]; }

const std::vector<MapPointMatch>& Frame::map_matches() const { return m_map_matches; }

std::vector<size_t> Frame::features_in_region(const Eigen::Vector2f& uv, float radius) const
{
    return m_kd_tree.radius_search(uv, radius);
}

} // namespace slam
