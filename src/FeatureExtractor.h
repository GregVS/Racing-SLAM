#pragma once

#include <opencv2/opencv.hpp>

struct ExtractedFeatures {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

namespace slam {

class FeatureExtractor {
  public:
    FeatureExtractor();

    ExtractedFeatures extract_features(const cv::Mat &image);
};

}; // namespace slam