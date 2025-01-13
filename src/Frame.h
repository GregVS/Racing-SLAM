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
    void remove_map_match(const MapPointMatch& match);
    void set_pose(const Eigen::Matrix4f& pose);

    const ExtractedFeatures& features() const;
    const cv::Mat descriptor(size_t index) const;
    const cv::KeyPoint& keypoint(size_t index) const;
    const MapPoint& map_match(size_t index) const;
    const Eigen::Matrix4f& pose() const;

    std::vector<size_t> features_in_region(const Eigen::Vector2f& uv, float radius) const;
    bool is_matched(size_t keypoint_index) const;

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

    MapPointIterator map_matches() const;

  private:
    size_t m_index;
    cv::Mat m_image;
    Eigen::Matrix4f m_pose;

    ExtractedFeatures m_features;
    KDTree2D m_kd_tree;

    std::vector<const MapPoint*> m_map_matches;
};

} // namespace slam
