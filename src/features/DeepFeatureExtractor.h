#pragma once

#include "FeatureExtractor.h"

namespace slam::features {

class DeepFeatureExtractor : public BaseFeatureExtractor {
  public:
    ExtractedFeatures extract_features(const cv::Mat& image, const cv::Mat& mask) const override;

  protected:
    float max_distance() const override
    {
        return 0.7;
    }
    cv::NormTypes norm_type() const override
    {
        return cv::NORM_L2;
    }
};

} // namespace slam::features
