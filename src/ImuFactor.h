#pragma once

#include <Eigen/Dense>

#include "Imu.h"

namespace ceres {
class CostFunction;
} // namespace ceres

namespace slam::optimization {

ceres::CostFunction* imu_preintegration(const imu::Preintegrated& delta, const Eigen::Vector3d& gravity);

ceres::CostFunction* imu_bias_random_walk(double duration, const imu::NoiseDensity& noise);

} // namespace slam::optimization
