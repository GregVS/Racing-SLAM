#include "ImuFactor.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace slam::optimization {

namespace {

Eigen::Matrix<double, 9, 9> whitener(const Eigen::Matrix<double, 9, 9>& covariance)
{
    const Eigen::LLT<Eigen::Matrix<double, 9, 9>> decomposition(covariance);
    if (decomposition.info() != Eigen::Success) {
        return Eigen::Matrix<double, 9, 9>::Identity();
    }
    return decomposition.matrixL().solve(Eigen::Matrix<double, 9, 9>::Identity());
}

class PreintegrationError {
  public:
    PreintegrationError(const imu::Preintegrated& delta, const Eigen::Vector3d& gravity)
        : m_delta(delta), m_gravity(gravity), m_whitener(whitener(delta.covariance))
    {
    }

    template <typename T>
    bool operator()(const T* const pose_i,
                    const T* const velocity_i,
                    const T* const bias_i,
                    const T* const pose_j,
                    const T* const velocity_j,
                    T* residuals) const
    {
        using Vector3 = Eigen::Matrix<T, 3, 1>;
        using Matrix3 = Eigen::Matrix<T, 3, 3>;

        const Eigen::Map<const Vector3> centre_i(pose_i + 3);
        const Eigen::Map<const Vector3> centre_j(pose_j + 3);
        const Eigen::Map<const Vector3> speed_i(velocity_i);
        const Eigen::Map<const Vector3> speed_j(velocity_j);

        // Convert to camera frame. This is backwards from the paper
        Matrix3 world_to_camera_i;
        Matrix3 world_to_camera_j;
        ceres::AngleAxisToRotationMatrix(pose_i, world_to_camera_i.data());
        ceres::AngleAxisToRotationMatrix(pose_j, world_to_camera_j.data());

        // Compute bias delta since preintegration was run
        Eigen::Matrix<T, 6, 1> bias_change;
        for (int i = 0; i < 3; i++) {
            bias_change[i] = bias_i[i] - T(m_delta.bias.gyro[i]);
            bias_change[i + 3] = bias_i[i + 3] - T(m_delta.bias.accel[i]);
        }
        const Eigen::Matrix<T, 9, 1> correction = m_delta.bias_jacobian.cast<T>() * bias_change;

        Matrix3 rotation_correction;
        ceres::AngleAxisToRotationMatrix(correction.template segment<3>(0).eval().data(), rotation_correction.data());
        const Matrix3 measured_rotation = m_delta.rotation.cast<T>() * rotation_correction;
        const Vector3 measured_velocity = m_delta.velocity.cast<T>() + correction.template segment<3>(3);
        const Vector3 measured_position = m_delta.position.cast<T>() + correction.template segment<3>(6);

        const T duration(m_delta.duration);
        const Vector3 gravity = m_gravity.cast<T>();

        // Predicted motion. Gravity has to be included because it is not part of the preintegration.
        const Matrix3 state_rotation = world_to_camera_i * world_to_camera_j.transpose();
        const Vector3 state_velocity = world_to_camera_i * (speed_j - speed_i - gravity * duration);
        const Vector3 state_position =
            world_to_camera_i * (centre_j - centre_i - speed_i * duration - T(0.5) * gravity * duration * duration);

        Eigen::Matrix<T, 9, 1> residual;
        const Matrix3 rotation_error = measured_rotation.transpose() * state_rotation;
        ceres::RotationMatrixToAngleAxis(rotation_error.data(), residual.data());
        residual.template segment<3>(3) = state_velocity - measured_velocity;
        residual.template segment<3>(6) = state_position - measured_position;

        // Standardize using covariance
        Eigen::Map<Eigen::Matrix<T, 9, 1>> whitened(residuals);
        whitened = m_whitener.cast<T>() * residual;
        return true;
    }

  private:
    const imu::Preintegrated m_delta;
    const Eigen::Vector3d m_gravity;
    const Eigen::Matrix<double, 9, 9> m_whitener;
};

class BiasRandomWalk {
  public:
    BiasRandomWalk(double gyro_sigma, double accel_sigma) : m_gyro_sigma(gyro_sigma), m_accel_sigma(accel_sigma) {}

    template <typename T> bool operator()(const T* const bias_i, const T* const bias_j, T* residuals) const
    {
        for (int i = 0; i < 3; i++) {
            residuals[i] = (bias_j[i] - bias_i[i]) / T(m_gyro_sigma);
            residuals[i + 3] = (bias_j[i + 3] - bias_i[i + 3]) / T(m_accel_sigma);
        }
        return true;
    }

  private:
    const double m_gyro_sigma;
    const double m_accel_sigma;
};

} // namespace

ceres::CostFunction* imu_preintegration(const imu::Preintegrated& delta, const Eigen::Vector3d& gravity)
{
    return new ceres::AutoDiffCostFunction<PreintegrationError, 9, 6, 3, 6, 6, 3>(
        new PreintegrationError(delta, gravity));
}

ceres::CostFunction* imu_bias_random_walk(double duration, const imu::NoiseDensity& noise)
{
    const double elapsed = std::sqrt(std::max(duration, 1e-9));
    return new ceres::AutoDiffCostFunction<BiasRandomWalk, 6, 6, 6>(
        new BiasRandomWalk(noise.gyro_bias * elapsed, noise.accel_bias * elapsed));
}

} // namespace slam::optimization
