#pragma once

#include <g2o/core/sparse_optimizer.h>
#include <Eigen/Core>
#include "Map.h"

namespace slam
{

class BundleAdjustment {
public:
    BundleAdjustment();

    void optimize_map(Map &map);

private:
    g2o::SparseOptimizer m_optimizer;

    void add_camera_pose(const int id, const Eigen::Isometry3d &pose, bool fixed = false);

    void add_point(const int id, const Eigen::Vector3d &point);

    void add_projection_edge(const int pose_id, const int point_id, const Eigen::Vector2d &measurement,
                             const Eigen::Matrix2d &information);

    void optimize(int iterations = 10);

    Eigen::Isometry3d get_camera_pose(const int id);

    Eigen::Vector3d get_point(const int id);
};

}