#pragma once

#include <Eigen/Dense>

namespace slam {
// Forward Declarations
class Map;
class Frame;
class Camera;
} // namespace slam

namespace slam::optimization {

struct FrameConfig {
    bool optimize;
    Frame* frame;
};

/** Ties the distance between two poses to wheel odometry measurement */
struct StepConstraint {
    const Frame* a;
    const Frame* b;
    float distance;
};

struct OptimizationConfig {
    bool optimize_points;
    const std::vector<FrameConfig> frames;
    std::vector<StepConstraint> step_constraints;
};

/** This function modifies the frames and map in place */
void optimize(const OptimizationConfig& config, const Camera& camera, Map& map);

} // namespace slam::optimization