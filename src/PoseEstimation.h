#pragma once

#include <Eigen/Dense>
#include <vector>

#include "Camera.h"
#include "Features.h"

namespace slam::pose {

struct PoseEstimate {
    Eigen::Matrix4f pose;
    std::vector<FeatureMatch> inlier_matches; // Matches that were used to estimate the pose
};

/* Returns the relative pose - change of basis from prev_features to features */
PoseEstimate estimate_pose(const ExtractedFeatures& prev_features,
                           const ExtractedFeatures& features,
                           const std::vector<FeatureMatch>& matches,
                           const Camera& camera);

} // namespace slam::pose