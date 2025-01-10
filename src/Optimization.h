#pragma once

#include <Eigen/Dense>

namespace slam {
// Forward Declarations
class Map;
class Frame;
class Camera;
} // namespace slam

namespace slam::optimization {

Eigen::Matrix4f optimize_pose(const Eigen::Matrix4f& pose, const Map& map, const Frame& frame, const Camera& camera);

} // namespace slam::optimization