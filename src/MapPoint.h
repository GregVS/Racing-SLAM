#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <unordered_map>

namespace slam {

class Frame;

class MapPoint {
  public:
    MapPoint(const Eigen::Vector3f& position);

    const Eigen::Vector3f& position() const;
    void set_position(const Eigen::Vector3f& position);

    void add_observation(const Frame* key_frame, size_t index);
    const std::unordered_map<const Frame*, size_t>& observations() const;

    const cv::Vec3b& color() const;
    void set_color(const cv::Vec3b& color);

  private:
    Eigen::Vector3f m_position;
    cv::Vec3b m_color;
    std::unordered_map<const Frame*, size_t> m_observations;
};

}  // namespace slam
