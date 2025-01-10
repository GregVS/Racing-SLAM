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

Eigen::Matrix4f
optimize_pose(const Eigen::Matrix4f& pose, const Map& map, const Frame& frame, const Camera& camera)
{
    auto problem = ceres::Problem();

    // Add pose parameters
    std::array<double, 6> pose_params;
    {
        auto rvec = matrix_to_rodrigues(pose.block<3, 3>(0, 0));
        auto tvec = pose.block<3, 1>(0, 3);
        pose_params[0] = rvec[0]; // Rotation
        pose_params[1] = rvec[1];
        pose_params[2] = rvec[2];
        pose_params[3] = tvec[0]; // Translation
        pose_params[4] = tvec[1];
        pose_params[5] = tvec[2];
    }

    // Add map point parameters
    std::vector<std::array<double, 3>> map_point_params;
    for (const auto& match : frame.map_matches()) {
        map_point_params.push_back(
            {match.point.position().x(), match.point.position().y(), match.point.position().z()});
    }

    // Setup problem
    for (int i = 0; i < map_point_params.size(); i++) {
        auto match = frame.map_matches()[i];
        auto cost_function = ReprojectionError::Create(frame.keypoint(match.keypoint_index).pt.x,
                                                       frame.keypoint(match.keypoint_index).pt.y,
                                                       camera.get_intrinsic_matrix()(0, 0),
                                                       camera.get_intrinsic_matrix()(0, 2),
                                                       camera.get_intrinsic_matrix()(1, 2));
        problem.AddResidualBlock(cost_function,
                                 new ceres::HuberLoss(sqrt(5.991)),
                                 pose_params.data(),
                                 map_point_params[i].data());
    }

    // Solve problem
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Extract optimized pose
    Eigen::Matrix4f optimized_pose = Eigen::Matrix4f::Identity();
    {
        auto rvec = Eigen::Vector3f(pose_params[0], pose_params[1], pose_params[2]);
        auto tvec = Eigen::Vector3f(pose_params[3], pose_params[4], pose_params[5]);
        optimized_pose.block<3, 3>(0, 0) = rodrigues_to_matrix(rvec);
        optimized_pose.block<3, 1>(0, 3) = tvec;
    }

    return optimized_pose;
}

} // namespace slam::optimization