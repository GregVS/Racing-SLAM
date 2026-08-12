#include "Imu.h"

#include <cmath>

namespace slam::imu {

namespace {

Eigen::Matrix3d skew(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d m;
    m << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
    return m;
}

// Uses the average of the angular velocities
Eigen::Matrix3d rotation_increment(const Sample& start, const Sample& end)
{
    return exp_so3(0.5 * (start.gyro + end.gyro) * (end.time - start.time));
}

} // namespace

// Axis-angle to rotation matrix
Eigen::Matrix3d exp_so3(const Eigen::Vector3d& rotation_vector)
{
    const double angle = rotation_vector.norm();
    const Eigen::Matrix3d generator = skew(rotation_vector);

    // Taylor expansion
    double sine_term = 0.0;
    double cosine_term = 0.0;
    if (angle < 1e-8) {
        sine_term = 1.0 - angle * angle / 6.0;
        cosine_term = 0.5 - angle * angle / 24.0;
    } else {
        sine_term = std::sin(angle) / angle;
        cosine_term = (1.0 - std::cos(angle)) / (angle * angle);
    }
    return Eigen::Matrix3d::Identity() + sine_term * generator + cosine_term * generator * generator;
}

// Inverse of exp_so3
Eigen::Vector3d log_so3(const Eigen::Matrix3d& rotation)
{
    const Eigen::AngleAxisd angle_axis(rotation);
    return angle_axis.angle() * angle_axis.axis();
}

// Derivative of exp_so3
Eigen::Matrix3d right_jacobian_so3(const Eigen::Vector3d& rotation_vector)
{
    const double angle = rotation_vector.norm();
    const Eigen::Matrix3d generator = skew(rotation_vector);
    if (angle < 1e-8) {
        return Eigen::Matrix3d::Identity() - 0.5 * generator + generator * generator / 6.0;
    }
    return Eigen::Matrix3d::Identity() - (1.0 - std::cos(angle)) / (angle * angle) * generator +
           (angle - std::sin(angle)) / (angle * angle * angle) * generator * generator;
}

Eigen::Matrix3d integrate_rotation(const std::vector<Sample>& samples)
{
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    for (size_t i = 0; i + 1 < samples.size(); i++) {
        rotation = rotation * rotation_increment(samples[i], samples[i + 1]);
    }
    return rotation;
}

Preintegrated preintegrate(const std::vector<Sample>& samples, const NoiseDensity& noise, const Bias& bias)
{
    if (samples.size() < 2) {
        return {};
    }

    // Remove bias from samples
    std::vector<Sample> corrected = samples;
    for (auto& sample : corrected) {
        sample.gyro -= bias.gyro;
        sample.accel -= bias.accel;
    }

    const State state = integrate(corrected, {}, Eigen::Vector3d::Zero());
    Preintegrated delta;
    delta.duration = corrected.back().time - corrected.front().time;
    delta.rotation = state.rotation;
    delta.velocity = state.velocity;
    delta.position = state.position;
    delta.bias = bias;

    Eigen::Matrix3d accumulated = Eigen::Matrix3d::Identity();
    for (size_t i = 0; i + 1 < corrected.size(); i++) {
        const double interval = corrected[i + 1].time - corrected[i].time;
        const Eigen::Vector3d w = 0.5 * (corrected[i].gyro + corrected[i + 1].gyro);
        const Eigen::Matrix3d R = exp_so3(w * interval);
        const Eigen::Vector3d a = 0.5 * (corrected[i].accel + R * corrected[i + 1].accel);

        // Jacobian of the transition function (see Forester's paper)
        Eigen::Matrix<double, 9, 9> trans_jacobian = Eigen::Matrix<double, 9, 9>::Identity();
        trans_jacobian.block<3, 3>(0, 0) = R.transpose();
        trans_jacobian.block<3, 3>(3, 0) = -accumulated * skew(a) * interval;
        trans_jacobian.block<3, 3>(6, 0) = -0.5 * accumulated * skew(a) * interval * interval;
        trans_jacobian.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * interval;

        // Jacobian for the function for a single sample
        Eigen::Matrix<double, 9, 6> sample_jacobian = Eigen::Matrix<double, 9, 6>::Zero();
        sample_jacobian.block<3, 3>(0, 0) = right_jacobian_so3(w * interval) * interval;
        sample_jacobian.block<3, 3>(3, 3) = accumulated * interval;
        sample_jacobian.block<3, 3>(6, 3) = 0.5 * accumulated * interval * interval;

        Eigen::Matrix<double, 6, 6> sample_covariance = Eigen::Matrix<double, 6, 6>::Zero();
        sample_covariance.diagonal().head<3>().setConstant(noise.gyro * noise.gyro / interval);
        sample_covariance.diagonal().tail<3>().setConstant(noise.accel * noise.accel / interval);

        delta.covariance = trans_jacobian * delta.covariance * trans_jacobian.transpose() +
                           sample_jacobian * sample_covariance * sample_jacobian.transpose();

        // Jacobian we use later to adjust for bias during optimization
        Eigen::Matrix<double, 9, 6> bias_jacobian = sample_jacobian;
        const Eigen::Matrix3d mean_rotation = 0.5 * (accumulated + accumulated * R);
        bias_jacobian.block<3, 3>(3, 3) = mean_rotation * interval;
        bias_jacobian.block<3, 3>(6, 3) = 0.5 * mean_rotation * interval * interval;

        const Eigen::Matrix3d trailing =
            accumulated * R * skew(corrected[i + 1].accel) * right_jacobian_so3(w * interval);
        bias_jacobian.block<3, 3>(3, 0) = -0.5 * trailing * interval * interval;
        bias_jacobian.block<3, 3>(6, 0) = -0.25 * trailing * interval * interval * interval;

        delta.bias_jacobian = trans_jacobian * delta.bias_jacobian - bias_jacobian;
        accumulated = accumulated * R;
    }
    return delta;
}

State predict(const State& initial, const Preintegrated& delta, const Eigen::Vector3d& gravity)
{
    State state;
    state.rotation = initial.rotation * delta.rotation;
    state.velocity = initial.velocity + gravity * delta.duration + initial.rotation * delta.velocity;
    state.position = initial.position + initial.velocity * delta.duration +
                     0.5 * gravity * delta.duration * delta.duration + initial.rotation * delta.position;
    return state;
}

State integrate(const std::vector<Sample>& samples, const State& initial, const Eigen::Vector3d& gravity)
{
    State state = initial;
    for (size_t i = 0; i + 1 < samples.size(); i++) {
        const double interval = samples[i + 1].time - samples[i].time;
        const Eigen::Matrix3d next_rotation = state.rotation * rotation_increment(samples[i], samples[i + 1]);

        // Midpoint
        const Eigen::Vector3d acceleration =
            0.5 * (state.rotation * samples[i].accel + next_rotation * samples[i + 1].accel) + gravity;

        state.position += state.velocity * interval + 0.5 * acceleration * interval * interval;
        state.velocity += acceleration * interval;
        state.rotation = next_rotation;
    }
    return state;
}

} // namespace slam::imu
