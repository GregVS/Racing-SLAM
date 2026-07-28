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

  private:
    cv::Ptr<cv::Feature2D> m_detector = cv::GFTTDetector::create(3000, 0.005, 7);
    cv::Ptr<cv::Feature2D> m_descriptor = cv::ORB::create();
};

} // namespace slam::features
