#pragma once

#include <Eigen/Dense>
#include "Features.h"

namespace slam {

class KeyFrame {
  public:
    KeyFrame(size_t index, const Eigen::Matrix4f& pose, const ExtractedFeatures& features);

    size_t index() const;
    const Eigen::Matrix4f& pose() const;
    const ExtractedFeatures& features() const;
    const cv::Mat descriptor(size_t index) const;

  private:
    size_t m_index;
    Eigen::Matrix4f m_pose;
    ExtractedFeatures m_features;
};

} // namespace slam