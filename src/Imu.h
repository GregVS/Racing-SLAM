#pragma once

#include <Eigen/Dense>
#include <vector>

namespace slam::imu {

struct Sample {
    double time;
    Eigen::Vector3d gyro;
    Eigen::Vector3d accel;
};

struct State {
    // Written in world coordinates
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
};

struct Bias {
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel = Eigen::Vector3d::Zero();
};

inline const Eigen::Vector3d GRAVITY_ENU{0.0, 0.0, -9.80665};

State integrate(const std::vector<Sample>& samples, const State& initial, const Eigen::Vector3d& gravity);

struct Preintegrated {
    double duration = 0.0;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 9, 9> covariance = Eigen::Matrix<double, 9, 9>::Zero();

    // Bias used when preintegration was run
    Bias bias;
    Eigen::Matrix<double, 9, 6> bias_jacobian = Eigen::Matrix<double, 9, 6>::Zero();
};

struct NoiseDensity {
    // Noise densities use specs for the Bosch BMI088. Overridden by IMU config
    double gyro = 2.44e-4;  // rad/s/sqrt(Hz)
    double accel = 1.86e-3; // m/s^2/sqrt(Hz)

    // Bias estimates are from EuRoC ADIS16448 configuration
    double gyro_bias = 2.78e-5;  // rad/s^2/sqrt(Hz)
    double accel_bias = 2.79e-3; // m/s^3/sqrt(Hz)
};

Eigen::Matrix3d integrate_rotation(const std::vector<Sample>& samples);

/** Forester's preintegration */
Preintegrated preintegrate(const std::vector<Sample>& samples, const NoiseDensity& noise, const Bias& bias = {});

/** Apply preintegrated state */
State predict(const State& initial, const Preintegrated& delta, const Eigen::Vector3d& gravity);

Eigen::Matrix3d exp_so3(const Eigen::Vector3d& rotation_vector);
Eigen::Vector3d log_so3(const Eigen::Matrix3d& rotation);
Eigen::Matrix3d right_jacobian_so3(const Eigen::Vector3d& rotation_vector);

} // namespace slam::imu
