#include "MotionModel.h"

#include <algorithm>
#include <cmath>

namespace slam::motion {

Eigen::Vector3f camera_center(const Eigen::Matrix4f& pose)
{
    return -pose.block<3, 3>(0, 0).transpose() * pose.block<3, 1>(0, 3);
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

} // namespace slam::motion
