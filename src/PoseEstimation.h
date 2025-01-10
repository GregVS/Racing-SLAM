#pragma once

#include <Eigen/Dense>
#include <vector>

#include "Camera.h"
#include "Features.h"

namespace slam::pose {

struct PoseEstimate {
    Eigen::Matrix4f pose;
    std::vector<FeatureMatch> inlier_matches;
};

PoseEstimate estimate_pose(const std::vector<FeatureMatch>& matches,
                           const ExtractedFeatures& prev_features,
                           const ExtractedFeatures& features,
                           const Camera& camera);

} // namespace slam::pose