#include "Trajectory.h"

#include <fstream>
#include <iostream>

namespace slam::trajectory {

bool write_kitti(const std::string& filename, const std::vector<Eigen::Matrix4f>& poses)
{
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Failed to open trajectory file: " << filename << std::endl;
        return false;
    }

    for (const auto& pose : poses) {
        // KITTI ground truth is camera-to-world, our poses are world-to-camera
        Eigen::Matrix3f rotation = pose.block<3, 3>(0, 0).transpose();
        Eigen::Vector3f translation = -rotation * pose.block<3, 1>(0, 3);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                file << rotation(i, j) << " ";
            }
            file << translation(i);
            file << (i < 2 ? " " : "\n");
        }
    }

    std::cout << "Wrote " << poses.size() << " poses to " << filename << std::endl;
    return true;
}

} // namespace slam::trajectory
