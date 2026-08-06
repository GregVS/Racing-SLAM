#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <unordered_map>

namespace slam {

class KeyFrame;

class MapPoint {
  public:
    MapPoint(const Eigen::Vector3f& position);

    const Eigen::Vector3f& position() const;
    void set_position(const Eigen::Vector3f& position);
    Eigen::Vector3f avg_viewing_normal() const;

    const std::unordered_map<KeyFrame*, size_t>& observations() const;
    bool is_observed_by(KeyFrame* key_frame) const;

    const cv::Vec3b& color() const;
    void set_color(const cv::Vec3b& color);

    bool track_consistent() const;
    void set_track_consistent();

  private:
    friend class Map;
    void add_observation(KeyFrame* key_frame, size_t index);
    void remove_observation(KeyFrame* key_frame);

    Eigen::Vector3f m_position;
    bool m_track_consistent = false;
    cv::Vec3b m_color;
    std::unordered_map<KeyFrame*, size_t> m_observations;
};

} // namespace slam
