#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "Features.h"
#include "KDTree.h"

namespace slam {

class Frame {
  public:
    Frame(int index, const cv::Mat& image, const ExtractedFeatures& features);

    size_t index() const;
    const cv::Mat& image() const;

    void add_map_match(const MapPointMatch& match);

    const ExtractedFeatures& features() const;
    const cv::Mat descriptor(size_t index) const;
    const cv::KeyPoint& keypoint(size_t index) const;

    std::vector<size_t> features_in_region(const Eigen::Vector2f& uv, float radius) const;
    const std::vector<MapPointMatch>& map_matches() const;

  private:
    size_t m_index;
    cv::Mat m_image;

    ExtractedFeatures m_features;
    KDTree2D m_kd_tree;

    std::vector<MapPointMatch> m_map_matches;
};

} // namespace slam