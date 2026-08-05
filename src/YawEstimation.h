#pragma once

#include <cstddef>
#include <vector>

#include "Camera.h"
#include "features/FeatureExtractor.h"

namespace slam::yaw {

struct YawEstimate {
    bool healthy = false;
    float radians = 0.0F;
    size_t inliers = 0;
    float inlier_ratio = 0.0F;
    float image_coverage = 0.0F;
    float robust_cost = 0.0F;
};

/** Estimates yaw change whose essential matrix best explains the tracked correspondences. */
YawEstimate estimate(const ExtractedFeatures& previous,
                     const ExtractedFeatures& current,
                     const std::vector<FeatureMatch>& matches,
                     const Camera& camera,
                     float max_yaw_radians);

} // namespace slam::yaw
