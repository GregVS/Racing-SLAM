#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

namespace slam {

// Forward declarations
class Frame;
class Camera;
class Map;
class MapPoint;

struct ExtractedFeatures {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

struct FeatureMatch {
    FeatureMatch(int train_index, int query_index)
        : train_index(train_index), query_index(query_index)
    {
    }

    size_t train_index;
    size_t query_index;
};

struct FilteredMatches {
    std::vector<FeatureMatch> matches;
    cv::Mat essential_matrix;
};

struct MapPointMatch {
    const MapPoint& point;
    size_t keypoint_index;
};
} // namespace slam

namespace slam::features {

class BaseFeatureExtractor {
  public:
    virtual ExtractedFeatures extract_features(const cv::Mat& image, const cv::Mat& mask) const = 0;

    virtual std::vector<FeatureMatch> match_features(const ExtractedFeatures& prev_features,
                                                     const ExtractedFeatures& features) const;

    virtual std::vector<MapPointMatch>
    match_features(const Frame& frame,
                   const Camera& camera,
                   const Map& map,
                   std::function<bool(const MapPoint&)> point_filter) const;

  protected:
    virtual float max_distance() const = 0;
    virtual cv::NormTypes norm_type() const = 0;
};

std::vector<FeatureMatch> unmatched_features(const Frame& frame1,
                                             const Frame& frame2,
                                             const std::vector<FeatureMatch>& matches);

} // namespace slam::features
