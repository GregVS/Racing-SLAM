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

KeyFrame::KeyFrame(Frame&& frame) : Frame(std::move(frame)) {}

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

Eigen::Vector3f Frame::camera_center() const
{
    return -m_pose.block<3, 3>(0, 0).transpose() * m_pose.block<3, 1>(0, 3);
}

void Frame::set_pose(const Eigen::Matrix4f& pose)
{
    m_pose = pose;
}

const InertialState& Frame::inertial() const
{
    return m_inertial;
}

void Frame::set_inertial(const InertialState& inertial)
{
    m_inertial = inertial;
}

imu::State inertial_state(const Frame& frame)
{
    imu::State state;
    state.rotation = frame.pose().block<3, 3>(0, 0).transpose().cast<double>();
    state.position = frame.camera_center().cast<double>();
    state.velocity = frame.inertial().velocity;
    return state;
}

void set_inertial_state(Frame& frame, const imu::State& state)
{
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    pose.block<3, 3>(0, 0) = state.rotation.transpose().cast<float>();
    pose.block<3, 1>(0, 3) = -pose.block<3, 3>(0, 0) * state.position.cast<float>();
    frame.set_pose(pose);

    InertialState inertial = frame.inertial();
    inertial.velocity = state.velocity;
    frame.set_inertial(inertial);
}

void Frame::add_map_match(const MapPointMatch& match)
{
    MapPoint* previous = m_map_matches[match.keypoint_index];
    if (previous == &match.point) {
        return;
    }
    if (previous == nullptr) {
        m_num_map_matches++;
    } else {
        m_matched_map_points.erase(previous);
    }
    for (size_t i = 0; i < m_map_matches.size(); i++) {
        if (m_map_matches[i] != &match.point || i == match.keypoint_index) {
            continue;
        }
        m_map_matches[i] = nullptr;
        if (m_num_map_matches > 0) {
            m_num_map_matches--;
        }
    }
    m_map_matches[match.keypoint_index] = &match.point;
    m_matched_map_points.insert(&match.point);
}

void Frame::remove_map_match(const MapPointMatch& match)
{
    m_matched_map_points.erase(&match.point);
    for (size_t i = 0; i < m_map_matches.size(); i++) {
        if (m_map_matches[i] != &match.point) {
            continue;
        }
        m_map_matches[i] = nullptr;
        if (m_num_map_matches > 0) {
            m_num_map_matches--;
        }
    }
}

MapPoint& Frame::map_match(size_t index) const
{
    return *m_map_matches[index];
}

size_t Frame::num_map_matches() const
{
    return m_num_map_matches;
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