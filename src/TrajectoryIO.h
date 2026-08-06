#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace slam::trajectory {

/** Writes the trajectory to a file in KITTI odometry format */
bool write_kitti(const std::string& filename, const std::vector<Eigen::Matrix4f>& poses);

} // namespace slam::trajectory
