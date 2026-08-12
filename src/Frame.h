#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <unordered_set>

#include "Imu.h"
#include "KDTree.h"
#include "features/FeatureExtractor.h"

namespace slam {

struct InertialState {
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    imu::Bias bias;
};

class Frame {
  public:
    class MapPointIterator {
      public:
        MapPointIterator(const Frame& frame, size_t index);
        MapPointMatch operator*() const;
        MapPointIterator& operator++();
        bool operator!=(const MapPointIterator& other) const;

        MapPointIterator begin() const;
        MapPointIterator end() const;

      private:
        const Frame& m_frame;
        size_t m_index;
    };

    Frame(int index, const cv::Mat& image, const ExtractedFeatures& features);

    size_t index() const;
    const cv::Mat& image() const;

    const cv::Mat descriptor(size_t index) const;
    const cv::KeyPoint& keypoint(size_t index) const;
    std::vector<size_t> features_in_region(const Eigen::Vector2f& uv, float radius) const;
    const ExtractedFeatures& features() const;

    Eigen::Vector3f camera_center() const;
    const Eigen::Matrix4f& pose() const;
    void set_pose(const Eigen::Matrix4f& pose);

    const InertialState& inertial() const;
    void set_inertial(const InertialState& inertial);

    void add_map_match(const MapPointMatch& match);
    MapPoint& map_match(size_t index) const;
    bool is_matched(size_t keypoint_index) const;
    bool is_matched(const MapPoint& point) const;
    size_t num_map_matches() const;
    MapPointIterator map_matches() const;

  private:
    friend class Map;
    void remove_map_match(const MapPointMatch& match);

    size_t m_index;
    cv::Mat m_image;
    Eigen::Matrix4f m_pose;
    InertialState m_inertial;

    ExtractedFeatures m_features;
    KDTree2D m_kd_tree;

    std::vector<MapPoint*> m_map_matches;
    std::unordered_set<const MapPoint*> m_matched_map_points;
    size_t m_num_map_matches = 0;
};

/** A frame promoted into the map. Only key frames observe map points. */
class KeyFrame : public Frame {
  public:
    explicit KeyFrame(Frame&& frame);
};

imu::State inertial_state(const Frame& frame);

void set_inertial_state(Frame& frame, const imu::State& state);

} // namespace slam
