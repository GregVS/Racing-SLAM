#include "Optimization.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <thread>

#include "Camera.h"
#include "Frame.h"
#include "Map.h"

class ReprojectionError {
  public:
    ReprojectionError(const float obs_x,
                      const float obs_y,
                      float focal_length_x,
                      float focal_length_y,
                      float principal_point_x,
                      float principal_point_y)
        : m_obs_x(obs_x), m_obs_y(obs_y), m_focal_length_x(focal_length_x), m_focal_length_y(focal_length_y),
          m_principal_point_x(principal_point_x), m_principal_point_y(principal_point_y)
    {
    }

    // Camera parameters are angle-axis rotation followed by the camera center in world coordinates.
    template <typename T> bool operator()(const T* const camera, const T* const point, T* residuals) const
    {
        const T* camera_rotation = &camera[0];
        const T* camera_center = &camera[3];

        T centered[3] = {
            point[0] - camera_center[0],
            point[1] - camera_center[1],
            point[2] - camera_center[2],
        };
        T p[3];
        ceres::AngleAxisRotatePoint(camera_rotation, centered, p);

        residuals[0] = T(m_focal_length_x) * p[0] / p[2] + T(m_principal_point_x) - T(m_obs_x);
        residuals[1] = T(m_focal_length_y) * p[1] / p[2] + T(m_principal_point_y) - T(m_obs_y);

        return true;
    }

    static ceres::CostFunction* Create(float obs_x,
                                       float obs_y,
                                       float focal_length_x,
                                       float focal_length_y,
                                       float principal_point_x,
                                       float principal_point_y)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
            new ReprojectionError(obs_x, obs_y, focal_length_x, focal_length_y, principal_point_x, principal_point_y));
    }

  private:
    const float m_obs_x;
    const float m_obs_y;
    const float m_focal_length_x;
    const float m_focal_length_y;
    const float m_principal_point_x;
    const float m_principal_point_y;
};

static const size_t MIN_OBSERVATIONS_TO_OPTIMIZE = 2;

static Eigen::Matrix3f rodrigues_to_matrix(const Eigen::Vector3f& rvec)
{
    Eigen::Matrix3f R;
    ceres::AngleAxisToRotationMatrix(rvec.data(), R.data());
    return R;
}

static Eigen::Vector3f matrix_to_rodrigues(const Eigen::Matrix3f& R)
{
    Eigen::Vector3f rvec;
    ceres::RotationMatrixToAngleAxis(R.data(), rvec.data());
    return rvec;
}

namespace slam::optimization {

Snapshot::Snapshot(const OptimizationConfig& config, Map& map, bool include_points)
{
    m_poses.reserve(config.frames.size());
    for (const auto& frame_config : config.frames) {
        if (frame_config.optimize) {
            m_poses.push_back({frame_config.frame, frame_config.frame->pose()});
        }
    }
    if (include_points) {
        m_positions.reserve(map.size());
        for (auto& point : map) {
            m_positions.push_back({&point, point.position()});
        }
    }
}

void Snapshot::restore() const
{
    for (const auto& [frame, pose] : m_poses) {
        frame->set_pose(pose);
    }
    for (const auto& [point, position] : m_positions) {
        point->set_position(position);
    }
}

bool optimize(const OptimizationConfig& config, const Camera& camera, Map& map)
{
    auto problem = ceres::Problem();
    std::unordered_map<const Frame*, std::array<double, 6>> frame_params;
    std::unordered_map<const MapPoint*, std::array<double, 3>> map_point_params;

    // Add pose parameters
    for (const auto& frame_config : config.frames) {
        auto* frame = frame_config.frame;
        frame_params.emplace(frame, std::array<double, 6>{0, 0, 0, 0, 0, 0});
        auto rvec = matrix_to_rodrigues(frame->pose().block<3, 3>(0, 0));
        auto center = frame->camera_center();
        frame_params[frame][0] = rvec[0]; // Rotation
        frame_params[frame][1] = rvec[1];
        frame_params[frame][2] = rvec[2];
        frame_params[frame][3] = center[0]; // Camera center
        frame_params[frame][4] = center[1];
        frame_params[frame][5] = center[2];
    }

    // Add points observed by frames being optimized
    std::unordered_set<const Frame*> frames_to_optimize;
    std::unordered_set<const MapPoint*> points_to_optimize;
    for (const auto& frame_config : config.frames) {
        if (!frame_config.optimize) {
            continue;
        }
        frames_to_optimize.insert(frame_config.frame);

        for (auto match : frame_config.frame->map_matches()) {
            const auto& point = match.point;
            if (point.observations().size() < MIN_OBSERVATIONS_TO_OPTIMIZE) {
                continue;
            }
            map_point_params.emplace(
                &point, std::array<double, 3>{point.position().x(), point.position().y(), point.position().z()});
            if (config.optimize_points) {
                points_to_optimize.insert(&point);
            }
        }
    }

    // Setup problem
    for (const auto& frame_config : config.frames) {
        bool optimize_frame = frame_config.optimize;
        auto* frame = frame_config.frame;
        for (const auto& match : frame->map_matches()) {
            if (map_point_params.find(&match.point) == map_point_params.end()) {
                continue;
            }

            auto* cost_function = ReprojectionError::Create(frame->keypoint(match.keypoint_index).pt.x,
                                                            frame->keypoint(match.keypoint_index).pt.y,
                                                            camera.get_intrinsic_matrix()(0, 0),
                                                            camera.get_intrinsic_matrix()(1, 1),
                                                            camera.get_intrinsic_matrix()(0, 2),
                                                            camera.get_intrinsic_matrix()(1, 2));
            problem.AddResidualBlock(cost_function,
                                     new ceres::HuberLoss(sqrt(5.991)),
                                     frame_params[frame].data(),
                                     map_point_params[&match.point].data());
        }
    }

    // Mark frames and points that are not being optimized as constant
    for (auto& [frame, params] : frame_params) {
        if (frames_to_optimize.find(frame) == frames_to_optimize.end() && problem.HasParameterBlock(params.data())) {
            problem.SetParameterBlockConstant(params.data());
        }
    }
    for (auto& [point, params] : map_point_params) {
        if (points_to_optimize.find(point) == points_to_optimize.end() && problem.HasParameterBlock(params.data())) {
            problem.SetParameterBlockConstant(params.data());
        }
    }

    // Solve problem
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = 100;
    options.num_threads = 1;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    bool usable =
        summary.IsSolutionUsable() && std::isfinite(summary.final_cost) && summary.final_cost <= summary.initial_cost;
    if (!usable) {
        std::cout << "Optimization rejected, unusable or non-improving solution" << std::endl;
        return false;
    }

    // Extract optimized pose
    for (const auto& frame_config : config.frames) {
        auto* frame = frame_config.frame;
        if (frames_to_optimize.find(frame) == frames_to_optimize.end()) {
            continue;
        }

        auto rvec = Eigen::Vector3f(frame_params[frame][0], frame_params[frame][1], frame_params[frame][2]);
        auto center = Eigen::Vector3f(frame_params[frame][3], frame_params[frame][4], frame_params[frame][5]);
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        pose.block<3, 3>(0, 0) = rodrigues_to_matrix(rvec);
        pose.block<3, 1>(0, 3) = -pose.block<3, 3>(0, 0) * center;
        frame->set_pose(pose);
    }

    // Extract optimized map points
    for (auto& point : map) {
        if (points_to_optimize.find(&point) == points_to_optimize.end()) {
            continue;
        }

        point.set_position(
            Eigen::Vector3f(map_point_params[&point][0], map_point_params[&point][1], map_point_params[&point][2]));
    }
    return true;
}

} // namespace slam::optimization
