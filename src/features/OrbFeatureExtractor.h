#pragma once

#include "FeatureExtractor.h"

namespace slam::features {

class OrbFeatureExtractor : public BaseFeatureExtractor {
  public:
    ExtractedFeatures extract_features(const cv::Mat& image, const cv::Mat& mask) const override;

  protected:
    float max_distance() const override
    {
        return 64;
    }

    cv::NormTypes norm_type() const override
    {
        return cv::NORM_HAMMING;
    }
};

} // namespace slam::features
