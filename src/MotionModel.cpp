#include "MotionModel.h"

#include <algorithm>
#include <cmath>

namespace slam::motion {

Eigen::Vector3f camera_center(const Eigen::Matrix4f& pose)
{
    return -pose.block<3, 3>(0, 0).transpose() * pose.block<3, 1>(0, 3);
}

float metric_distance(const std::vector<float>& steps, size_t from, size_t to)
{
    float distance = 0;
    for (size_t i = from; i < to && i < steps.size(); i++) {
        distance += steps[i];
    }
    return distance;
}

float rotation_difference_degrees(const Eigen::Matrix4f& a, const Eigen::Matrix4f& b)
{
    Eigen::Matrix3f relative = a.block<3, 3>(0, 0) * b.block<3, 3>(0, 0).transpose();
    float cosine = std::clamp((relative.trace() - 1.0F) * 0.5F, -1.0F, 1.0F);
    return std::acos(cosine) * 180.0F / 3.14159265358979323846F;
}

bool is_rotation_plausible(const Eigen::Matrix4f& previous, const Eigen::Matrix4f& candidate, float elapsed_seconds)
{
    return elapsed_seconds > 0.0F &&
           rotation_difference_degrees(previous, candidate) <= MAX_ANGULAR_SPEED_DEGREES * elapsed_seconds;
}

Eigen::Matrix4f with_metric_step(const Eigen::Matrix4f& previous, const Eigen::Matrix4f& candidate, float distance)
{
    Eigen::Matrix4f result = candidate;
    Eigen::Vector3f previous_forward = previous.block<3, 3>(0, 0).transpose().col(2);
    Eigen::Vector3f candidate_forward = candidate.block<3, 3>(0, 0).transpose().col(2);
    Eigen::Vector3f direction = previous_forward + candidate_forward;
    if (direction.squaredNorm() < 1e-8F) {
        direction = previous_forward;
    }
    direction.normalize();

    Eigen::Vector3f center = camera_center(previous) + distance * direction;
    result.block<3, 1>(0, 3) = -result.block<3, 3>(0, 0) * center;
    return result;
}

} // namespace slam::motion
