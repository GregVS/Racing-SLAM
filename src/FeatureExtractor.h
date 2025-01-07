#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "Camera.h"
namespace slam {

struct ExtractedFeatures {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

struct FeatureMatch {
    FeatureMatch(int train_index, int query_index)
        : train_index(train_index), query_index(query_index)
    {
    }

    int train_index;
    int query_index;
};

struct FilteredMatches {
    std::vector<FeatureMatch> matches;
    cv::Mat essential_matrix;
};

class FeatureExtractor {
  public:
    FeatureExtractor();

    ExtractedFeatures extract_features(const cv::Mat &image,
                                       const cv::InputArray &mask = cv::noArray()) const;

    std::vector<FeatureMatch> match_features(const ExtractedFeatures &prev_features,
                                             const ExtractedFeatures &features) const;

    FilteredMatches filter_matches(const std::vector<FeatureMatch> &matches,
                                   const ExtractedFeatures &prev_features,
                                   const ExtractedFeatures &features,
                                   const Camera &camera) const;

  private:
    static constexpr int MAX_ORB_DISTANCE = 32;
};

}; // namespace slam
