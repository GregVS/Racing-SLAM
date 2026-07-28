#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace slam::trajectory {

/** Writes camera-to-world poses in KITTI odometry format: one frame per line, 12 space
 * separated values, the 3x4 pose matrix in row-major order. Takes world-to-camera poses and
 * inverts them. Returns false if the file could not be opened. */
bool write_kitti(const std::string& filename, const std::vector<Eigen::Matrix4f>& poses);

} // namespace slam::trajectory
