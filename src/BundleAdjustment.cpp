#include "BundleAdjustment.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>

namespace slam
{

Eigen::Vector2d cv_point_to_eigen(const cv::Point2f &point)
{
    return Eigen::Vector2d(point.x, point.y);
}

Eigen::Vector3d cv_point_to_eigen(const cv::Point3f &point)
{
    return Eigen::Vector3d(point.x, point.y, point.z);
}

cv::Point2f eigen_to_cv_point(const Eigen::Vector2d &vec)
{
    return cv::Point2f(vec.x(), vec.y());
}

cv::Point3f eigen_to_cv_point(const Eigen::Vector3d &vec)
{
    return cv::Point3f(vec.x(), vec.y(), vec.z());
}

Eigen::Isometry3d cv_mat_to_isometry3d(const cv::Mat &pose)
{
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    if (pose.rows < 3 || pose.cols < 4) {
        throw std::runtime_error("Invalid pose matrix dimensions");
    }

    Eigen::Matrix3d R;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R(i, j) = pose.at<double>(i, j);
        }
    }

    Eigen::Vector3d t;
    for (int i = 0; i < 3; i++) {
        t(i) = pose.at<double>(i, 3);
    }

    T.linear() = R;
    T.translation() = t;

    return T;
}

cv::Mat isometry3d_to_cv_mat(const Eigen::Isometry3d &T)
{
    cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            pose.at<double>(i, j) = T.linear()(i, j);
        }
    }

    for (int i = 0; i < 3; i++) {
        pose.at<double>(i, 3) = T.translation()(i);
    }

    return pose;
}

BundleAdjustment::BundleAdjustment()
{
    m_optimizer.setVerbose(true);

    auto linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType> >();
    auto block_solver = std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
    m_optimizer.setAlgorithm(algorithm);
}

void BundleAdjustment::optimize_map(Map &map)
{
    g2o::CameraParameters *cam = new g2o::CameraParameters(
            map.get_camera().get_intrinsic_matrix().at<double>(0, 0),
            Eigen::Vector2d(map.get_camera().get_width() / 2, map.get_camera().get_height() / 2), 0.0);
    cam->setId(0);
    m_optimizer.addParameter(cam);

    for (const auto &frame : map.get_frames()) {
        cv::Mat pose = frame->get_pose().inv();
        add_camera_pose(frame->get_id() * 2, cv_mat_to_isometry3d(pose), frame->get_id() < 2);
        std::cout << "Frame " << frame->get_id() << " pose: " << frame->get_pose().col(3).t() << std::endl;
    }

    for (const auto &[id, point] : map.get_map_points()) {
        add_point(point.get_id() * 2 + 1, cv_point_to_eigen(point.get_position()));

        point.for_each_observation([&](const Frame *frame, int keypoint_idx) {
            add_projection_edge(frame->get_id() * 2, point.get_id() * 2 + 1,
                                cv_point_to_eigen(frame->get_keypoint(keypoint_idx).pt), Eigen::Matrix2d::Identity());
        });
    }

    optimize();

    for (auto &frame : map.get_frames()) {
        cv::Mat pose = isometry3d_to_cv_mat(get_camera_pose(frame->get_id() * 2)).inv();
        frame->set_pose(pose);
        std::cout << "Frame " << frame->get_id() << " pose: " << pose.col(3).t() << std::endl;
    }

    for (auto &[id, point] : map.get_map_points()) {
        point.set_position(eigen_to_cv_point(get_point(point.get_id() * 2 + 1)));
    }

    m_optimizer.clear();
}

void BundleAdjustment::optimize(int iterations)
{
    m_optimizer.initializeOptimization();
    m_optimizer.optimize(iterations);
}

void BundleAdjustment::add_camera_pose(const int id, const Eigen::Isometry3d &pose, bool fixed)
{
    auto v = new g2o::VertexSE3Expmap();
    v->setId(id);
    v->setEstimate(g2o::SE3Quat(pose.rotation(), pose.translation()));
    v->setFixed(fixed);
    m_optimizer.addVertex(v);
}

void BundleAdjustment::add_point(const int id, const Eigen::Vector3d &point)
{
    auto v = new g2o::VertexPointXYZ();
    v->setId(id);
    v->setEstimate(point);
    v->setMarginalized(true);
    m_optimizer.addVertex(v);
}

void BundleAdjustment::add_projection_edge(const int pose_id, const int point_id, const Eigen::Vector2d &measurement,
                                           const Eigen::Matrix2d &information)
{
    auto edge = new g2o::EdgeProjectXYZ2UV();
    edge->setVertex(0, m_optimizer.vertex(point_id));
    edge->setVertex(1, m_optimizer.vertex(pose_id));
    edge->setMeasurement(measurement);
    edge->setInformation(information);
    edge->setParameterId(0, 0);
    edge->setRobustKernel(new g2o::RobustKernelHuber());
    m_optimizer.addEdge(edge);
}

Eigen::Isometry3d BundleAdjustment::get_camera_pose(const int id)
{
    auto v = dynamic_cast<g2o::VertexSE3Expmap *>(m_optimizer.vertex(id));
    if (v) {
        g2o::SE3Quat se3 = v->estimate();
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.rotate(se3.rotation());
        pose.translate(se3.translation());
        return pose;
    }
    return Eigen::Isometry3d::Identity();
}

Eigen::Vector3d BundleAdjustment::get_point(const int id)
{
    auto v = dynamic_cast<g2o::VertexPointXYZ *>(m_optimizer.vertex(id));
    if (v) {
        return v->estimate();
    }
    return Eigen::Vector3d::Zero();
}

};