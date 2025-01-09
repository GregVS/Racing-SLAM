#pragma once

#include <Eigen/Dense>
#include <vector>

#include "Camera.h"
#include "FeatureExtractor.h"

namespace slam {

struct PoseEstimate {
    Eigen::Matrix4f pose;
    std::vector<FeatureMatch> inlier_matches;
};

class PoseEstimator {
  public:
    PoseEstimator() {}

    PoseEstimate estimate_pose(const std::vector<FeatureMatch>& matches,
                                 const ExtractedFeatures& prev_features,
                                 const ExtractedFeatures& features,
                                 const Camera& camera) const;
};

} // namespace slam