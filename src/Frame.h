#pragma once

#include <opencv2/opencv.hpp>

#include "FeatureExtractor.h"

namespace slam {

class Frame {
  public:
    Frame(int index, const cv::Mat& image);

    int index() const;
    const cv::Mat& image() const;

    ExtractedFeatures set_features(const ExtractedFeatures& features);
    const ExtractedFeatures& features() const;

  private:
    const int m_index;
    const cv::Mat m_image;
    ExtractedFeatures m_features;
};

} // namespace slam