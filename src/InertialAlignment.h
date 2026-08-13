#pragma once

#include <Eigen/Dense>
#include <vector>

#include "Imu.h"

namespace slam::imu {

struct Alignment {
    bool valid = false;

    double scale = 1.0; // Multiply with map units to give meters
    double scale_uncertainty = 1.0;
    Eigen::Vector3d gravity = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> velocities; // Metric velocity per key frame

    double gravity_magnitude_error = 0.0;
    double gravity_uncertainty = 1.0; // Direction in radians
    double residual = 0.0;

    size_t triples = 0;
};

/** Recover scale and gravity using VINS-Mono strategy */
Alignment align(const std::vector<Eigen::Matrix3d>& rotations,
                const std::vector<Eigen::Vector3d>& positions,
                const std::vector<double>& times,
                const std::vector<Preintegrated>& summaries);

} // namespace slam::imu
