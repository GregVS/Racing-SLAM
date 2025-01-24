#pragma once

#include <Eigen/Dense>
#include <memory>
#include <optional>

#include "features/FeatureExtractor.h"
#include "Frame.h"

namespace slam::init {

struct Initialization {
    Frame ref_frame;
    Frame query_frame;
    std::vector<FeatureMatch> matches;
};

static constexpr int MAX_REF_CHANCES = 5;
static constexpr int MIN_TRIANGULATED_POINTS = 50;

std::optional<Initialization> find_initializing_frames(std::function<std::optional<Frame>()> next_frame,
                                                         const Camera& camera,
                                                         const features::BaseFeatureExtractor& feature_extractor);

} // namespace slam::init