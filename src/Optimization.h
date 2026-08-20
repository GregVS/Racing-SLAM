#pragma once

#include <memory>
#include <variant>
#include <vector>

#include <Eigen/Dense>

#include "Imu.h"
#include "ImuStream.h"

namespace slam {
// Forward Declarations
class Map;
class Frame;
class Camera;
class KeyFrame;
} // namespace slam

namespace slam::optimization {

struct FrameConfig {
    bool optimize;
    Frame* frame;
};

struct InertialInput {
    const imu::Stream* stream = nullptr; // nullptr before IMU alignment
    Eigen::Vector3d gravity = Eigen::Vector3d::Zero();
    imu::NoiseDensity noise;
    double seconds_per_frame = 0.0;

    double time_of(size_t frame_index) const
    {
        return static_cast<double>(frame_index) * seconds_per_frame;
    }
    double attitude_error_density = 2.76e-3; // should be overridden by the config

    bool usable() const
    {
        return stream != nullptr && seconds_per_frame > 0.0;
    }

    bool aligned() const
    {
        return stream != nullptr;
    }
};

struct RotationPrior {
    /** World to camera rotation */
    Eigen::Matrix3d predicted = Eigen::Matrix3d::Identity();
    double sigma_radians = 0.0;

    bool enabled() const
    {
        return sigma_radians > 0.0;
    }
};

struct InertialDelta {
    const Frame* previous = nullptr;
    imu::Preintegrated summary;
    Eigen::Vector3d gravity = Eigen::Vector3d::Zero();
    imu::NoiseDensity noise;

    bool enabled() const
    {
        return previous != nullptr && summary.duration > 0.0;
    }
};

using InertialConstraint = std::variant<std::monostate, RotationPrior, InertialDelta>;

bool refine_pose(Frame& frame, const Camera& camera, const InertialConstraint& inertial = {});

bool bundle_adjust(const std::vector<FrameConfig>& frames,
                   const Camera& camera,
                   Map& map,
                   const InertialInput& inertial = {});

struct PoseGraphConstraint {
    size_t from = 0;
    size_t to = 0;
    Eigen::Matrix4d relative = Eigen::Matrix4d::Identity();
};

/** PGO for poses and loop constraints. Moves keyframes and rigidly transforms their points */
bool pose_graph(const std::vector<std::shared_ptr<KeyFrame>>& key_frames,
                const std::vector<PoseGraphConstraint>& loops,
                Map& map,
                bool four_dof,
                const Eigen::Vector3d& gravity);

} // namespace slam::optimization
