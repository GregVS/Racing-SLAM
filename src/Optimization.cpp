#include "Optimization.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>

#include "Camera.h"
#include "Frame.h"
#include "Map.h"

class ReprojectionError {
  public:
    ReprojectionError(const float obs_x,
                      const float obs_y,
                      float focal_length,
                      float principal_point_x,
                      float principal_point_y)
        : m_obs_x(obs_x), m_obs_y(obs_y), m_focal_length(focal_length),
          m_principal_point_x(principal_point_x), m_principal_point_y(principal_point_y)
    {
    }

    // The camera params (angle/translation) should transform from world coords to camera coords
    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const
    {
        const T* camera_rotation = &camera[0];
        const T* camera_translation = &camera[3];

        T p[3];
        ceres::AngleAxisRotatePoint(camera_rotation, point, p);
        p[0] += camera_translation[0];
        p[1] += camera_translation[1];
        p[2] += camera_translation[2];

        T normalized_obs_x = (T(m_obs_x) - T(m_principal_point_x)) / T(m_focal_length);
        T normalized_obs_y = (T(m_obs_y) - T(m_principal_point_y)) / T(m_focal_length);

        residuals[0] = p[0] / p[2] - normalized_obs_x;
        residuals[1] = p[1] / p[2] - normalized_obs_y;

        return true;
    }

    static ceres::CostFunction* Create(float obs_x,
                                       float obs_y,
                                       float focal_length,
                                       float principal_point_x,
                                       float principal_point_y)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
            new ReprojectionError(obs_x,
                                  obs_y,
                                  focal_length,
                                  principal_point_x,
                                  principal_point_y));
    }

  private:
    const float m_obs_x;
    const float m_obs_y;
    const float m_focal_length;
    const float m_principal_point_x;
    const float m_principal_point_y;
};

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

void optimize(const OptimizationConfig& config, const Camera& camera, Map& map)
{
    auto problem = ceres::Problem();
    std::unordered_map<const Frame*, std::array<double, 6>> frame_params;
    std::unordered_map<const MapPoint*, std::array<double, 3>> map_point_params;

    // Add pose parameters
    for (const auto& [_, frame] : config.frames) {
        frame_params.emplace(frame, std::array<double, 6>{0, 0, 0, 0, 0, 0});
        auto rvec = matrix_to_rodrigues(frame->pose().block<3, 3>(0, 0));
        auto tvec = frame->pose().block<3, 1>(0, 3);
        frame_params[frame][0] = rvec[0]; // Rotation
        frame_params[frame][1] = rvec[1];
        frame_params[frame][2] = rvec[2];
        frame_params[frame][3] = tvec[0]; // Translation
        frame_params[frame][4] = tvec[1];
        frame_params[frame][5] = tvec[2];
    }

    // Add points
    std::unordered_set<const Frame*> frames_to_optimize;
    std::unordered_set<const MapPoint*> points_to_optimize;
    for (const auto& [optimize_frame, frame] : config.frames) {
        if (optimize_frame) {
            frames_to_optimize.insert(frame);

            for (auto match : frame->map_matches()) {
                const auto& point = match.point;
                map_point_params.emplace(&point,
                                         std::array<double, 3>{point.position().x(),
                                                               point.position().y(),
                                                               point.position().z()});
                if (config.optimize_points) {
                    points_to_optimize.insert(&point);
                }
            }
        }
    }

    // Setup problem
    for (const auto& [optimize_frame, frame] : config.frames) {
        for (const auto& match : frame->map_matches()) {
            if (map_point_params.find(&match.point) == map_point_params.end()) {
                continue;
            }

            auto cost_function = ReprojectionError::Create(
                frame->keypoint(match.keypoint_index).pt.x,
                frame->keypoint(match.keypoint_index).pt.y,
                camera.get_intrinsic_matrix()(0, 0),
                camera.get_intrinsic_matrix()(0, 2),
                camera.get_intrinsic_matrix()(1, 2));
            problem.AddResidualBlock(cost_function,
                                     new ceres::HuberLoss(sqrt(5.991)),
                                     frame_params[frame].data(),
                                     map_point_params[&match.point].data());

            if (points_to_optimize.find(&match.point) == points_to_optimize.end()) {
                problem.SetParameterBlockConstant(map_point_params[&match.point].data());
            }

            if (frames_to_optimize.find(frame) == frames_to_optimize.end()) {
                problem.SetParameterBlockConstant(frame_params[frame].data());
            }
        }
    }

    // Solve problem
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = 10;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    // Extract optimized pose
    for (const auto& [optimize, frame] : config.frames) {
        if (frames_to_optimize.find(frame) == frames_to_optimize.end()) {
            continue;
        }

        auto rvec = Eigen::Vector3f(frame_params[frame][0],
                                    frame_params[frame][1],
                                    frame_params[frame][2]);
        auto tvec = Eigen::Vector3f(frame_params[frame][3],
                                    frame_params[frame][4],
                                    frame_params[frame][5]);
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        pose.block<3, 3>(0, 0) = rodrigues_to_matrix(rvec);
        pose.block<3, 1>(0, 3) = tvec;
        frame->set_pose(pose);
    }

    // Extract optimized map points
    for (auto& point : map) {
        if (points_to_optimize.find(&point) == points_to_optimize.end()) {
            continue;
        }

        point.set_position(Eigen::Vector3f(map_point_params[&point][0],
                                           map_point_params[&point][1],
                                           map_point_params[&point][2]));
    }
}

} // namespace slam::optimization