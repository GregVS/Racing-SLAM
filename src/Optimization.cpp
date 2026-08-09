#include "Optimization.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cmath>
#include <opencv2/opencv.hpp>

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

namespace {

// Shared solver setup for both graphs
bool solve(ceres::Problem& problem)
{
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = 100;
    options.num_threads = 1;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << '\n';
    bool usable =
        summary.IsSolutionUsable() && std::isfinite(summary.final_cost) && summary.final_cost <= summary.initial_cost;
    if (!usable) {
        std::cout << "Optimization rejected, unusable or non-improving solution\n";
    }
    return usable;
}

std::array<double, 6> pack_pose(const Frame& frame)
{
    auto rvec = matrix_to_rodrigues(frame.pose().block<3, 3>(0, 0));
    auto center = frame.camera_center();
    return {rvec[0], rvec[1], rvec[2], center[0], center[1], center[2]};
}

void unpack_pose(const std::array<double, 6>& params, Frame& frame)
{
    auto rvec = Eigen::Vector3f(params[0], params[1], params[2]);
    auto center = Eigen::Vector3f(params[3], params[4], params[5]);
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    pose.block<3, 3>(0, 0) = rodrigues_to_matrix(rvec);
    pose.block<3, 1>(0, 3) = -pose.block<3, 3>(0, 0) * center;
    frame.set_pose(pose);
}

ceres::CostFunction* reprojection(const Frame& frame, size_t keypoint_index, const Camera& camera)
{
    return ReprojectionError::Create(frame.keypoint(keypoint_index).pt.x,
                                     frame.keypoint(keypoint_index).pt.y,
                                     camera.get_intrinsic_matrix()(0, 0),
                                     camera.get_intrinsic_matrix()(1, 1),
                                     camera.get_intrinsic_matrix()(0, 2),
                                     camera.get_intrinsic_matrix()(1, 2));
}

} // namespace

bool refine_pose(Frame& frame, const Camera& camera)
{
    auto problem = ceres::Problem();
    std::array<double, 6> pose_params = pack_pose(frame);
    std::unordered_map<const MapPoint*, std::array<double, 3>> point_params;

    for (auto match : frame.map_matches()) {
        const auto& point = match.point;
        if (point.observations().size() < MIN_OBSERVATIONS_TO_OPTIMIZE) {
            continue;
        }
        point_params.emplace(&point,
                             std::array<double, 3>{point.position().x(), point.position().y(), point.position().z()});
    }
    for (const auto& match : frame.map_matches()) {
        if (point_params.find(&match.point) == point_params.end()) {
            continue;
        }
        problem.AddResidualBlock(reprojection(frame, match.keypoint_index, camera),
                                 new ceres::HuberLoss(sqrt(5.991)),
                                 pose_params.data(),
                                 point_params[&match.point].data());
    }
    for (auto& [point, params] : point_params) {
        if (problem.HasParameterBlock(params.data())) {
            problem.SetParameterBlockConstant(params.data());
        }
    }

    if (!solve(problem)) {
        return false;
    }
    unpack_pose(pose_params, frame);
    return true;
}

bool bundle_adjust(const std::vector<FrameConfig>& frames, const Camera& camera, Map& map)
{
    auto problem = ceres::Problem();
    std::unordered_map<const Frame*, std::array<double, 6>> frame_params;
    std::unordered_map<const MapPoint*, std::array<double, 3>> map_point_params;

    // Add pose parameters
    for (const auto& frame_config : frames) {
        frame_params.emplace(frame_config.frame, pack_pose(*frame_config.frame));
    }

    // Points observed by the frames being optimized become free parameters
    std::unordered_set<const Frame*> frames_to_optimize;
    std::unordered_set<const MapPoint*> points_to_optimize;
    for (const auto& frame_config : frames) {
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
            points_to_optimize.insert(&point);
        }
    }

    for (const auto& frame_config : frames) {
        auto* frame = frame_config.frame;
        for (const auto& match : frame->map_matches()) {
            if (map_point_params.find(&match.point) == map_point_params.end()) {
                continue;
            }
            problem.AddResidualBlock(reprojection(*frame, match.keypoint_index, camera),
                                     new ceres::HuberLoss(sqrt(5.991)),
                                     frame_params[frame].data(),
                                     map_point_params[&match.point].data());
        }
    }
    for (auto& [frame, params] : frame_params) {
        if (frames_to_optimize.find(frame) == frames_to_optimize.end() && problem.HasParameterBlock(params.data())) {
            problem.SetParameterBlockConstant(params.data());
        }
    }

    if (!solve(problem)) {
        return false;
    }
    for (const auto& frame_config : frames) {
        if (frames_to_optimize.find(frame_config.frame) != frames_to_optimize.end()) {
            unpack_pose(frame_params[frame_config.frame], *frame_config.frame);
        }
    }
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
