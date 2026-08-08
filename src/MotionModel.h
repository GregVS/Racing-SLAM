#pragma once

#include <Eigen/Dense>

namespace slam::motion {

// More than twice the maximum angular speed in the valid racing benchmarks.
constexpr float MAX_ANGULAR_SPEED_DEGREES = 120.0F;

Eigen::Vector3f camera_center(const Eigen::Matrix4f& pose);

float rotation_difference_degrees(const Eigen::Matrix4f& a, const Eigen::Matrix4f& b);

bool is_rotation_plausible(const Eigen::Matrix4f& previous, const Eigen::Matrix4f& candidate, float elapsed_seconds);

} // namespace slam::motion
