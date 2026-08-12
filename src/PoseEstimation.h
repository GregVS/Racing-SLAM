#pragma once

#include <Eigen/Dense>
#include <vector>

#include "Camera.h"
#include "features/FeatureExtractor.h"

namespace slam::pose {

struct PoseEstimate {
    Eigen::Matrix4f pose;
    std::vector<FeatureMatch> inlier_matches; // Matches that were used to estimate the pose
};

/** Returns the relative pose - change of basis from prev_features to features */
PoseEstimate estimate_pose(const ExtractedFeatures& prev_features,
                           const ExtractedFeatures& features,
                           const std::vector<FeatureMatch>& matches,
                           const Camera& camera);

/** Same as estimate_pose but with the rotation provided */
PoseEstimate estimate_pose_with_known_rotation(const ExtractedFeatures& prev_features,
                                               const ExtractedFeatures& features,
                                               const std::vector<FeatureMatch>& matches,
                                               const Camera& camera,
                                               const Eigen::Matrix3f& rotation);

} // namespace slam::pose