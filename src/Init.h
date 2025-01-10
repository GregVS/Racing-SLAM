#pragma once

#include <Eigen/Dense>
#include <memory>
#include <optional>

#include "Features.h"
#include "Frame.h"

namespace slam::init {

struct InitializerResult {
    /* Note that the frames may be null if the initialization failed */
    std::shared_ptr<const Frame> ref_frame;
    std::shared_ptr<const Frame> query_frame;
    std::vector<FeatureMatch>    inlier_matches;
    Eigen::Matrix4f              pose;
};

static constexpr int   MIN_KEYPOINTS = 100;
static constexpr int   MIN_MATCHES = 100;
static constexpr int   MAX_REF_CHANCES = 5;
static constexpr float GOOD_MATCH_DISTANCE = 15.0;
static constexpr int   MIN_GOOD_MATCHES = 50;

InitializerResult find_initializing_frames(std::function<std::shared_ptr<const Frame>()> next_frame,
                                           const Camera&                                 camera);

} // namespace slam::init