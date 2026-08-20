#include "Optimization.h"

#include <algorithm>
#include <array>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <thread>
#include <unordered_map>

#include <Eigen/Geometry>

#include "Camera.h"
#include "Frame.h"
#include "ImuFactor.h"
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

//** Residuals between a predicted rotation and the current rotation */
class PredictedRotationError {
  public:
    PredictedRotationError(const Eigen::Matrix3d& predicted, double sigma) : m_predicted(predicted), m_sigma(sigma) {}

    template <typename T> bool operator()(const T* const camera, T* residuals) const
    {
        T current[9];
        ceres::AngleAxisToRotationMatrix(camera, current);
        const Eigen::Map<const Eigen::Matrix<T, 3, 3>> rotation(current);
        const Eigen::Matrix<T, 3, 3> difference = m_predicted.cast<T>().transpose() * rotation;
        T offset[3];
        ceres::RotationMatrixToAngleAxis(difference.data(), offset);
        residuals[0] = offset[0] / T(m_sigma);
        residuals[1] = offset[1] / T(m_sigma);
        residuals[2] = offset[2] / T(m_sigma);
        return true;
    }

  private:
    const Eigen::Matrix3d m_predicted;
    const double m_sigma;
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

constexpr int POSE_ITERATIONS = 10;
constexpr int BA_ITERATIONS = 10;
constexpr int PGO_ITERATIONS = 20;

int solver_threads()
{
    return static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
}

bool solve(ceres::Problem& problem, int max_iterations, ceres::LinearSolverType linear_solver)
{
    ceres::Solver::Options options;
    options.linear_solver_type = linear_solver;
    options.max_num_iterations = max_iterations;
    options.num_threads = solver_threads();
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

std::array<double, 3> pack_velocity(const Frame& frame)
{
    const auto& velocity = frame.inertial().velocity;
    return {velocity[0], velocity[1], velocity[2]};
}

std::array<double, 6> pack_bias(const Frame& frame)
{
    const auto& bias = frame.inertial().bias;
    return {bias.gyro[0], bias.gyro[1], bias.gyro[2], bias.accel[0], bias.accel[1], bias.accel[2]};
}

void unpack_inertial(const std::array<double, 3>& velocity, const std::array<double, 6>& bias, Frame& frame)
{
    InertialState state;
    state.velocity = Eigen::Vector3d(velocity[0], velocity[1], velocity[2]);
    state.bias.gyro = Eigen::Vector3d(bias[0], bias[1], bias[2]);
    state.bias.accel = Eigen::Vector3d(bias[3], bias[4], bias[5]);
    frame.set_inertial(state);
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

bool refine_pose(Frame& frame, const Camera& camera, const InertialConstraint& inertial)
{
    auto problem = ceres::Problem();
    std::array<double, 6> pose_params = pack_pose(frame);
    std::array<double, 6> previous_pose{};
    std::array<double, 3> previous_velocity{};
    std::array<double, 6> previous_bias{};
    std::array<double, 3> velocity_params{};
    std::unordered_map<const MapPoint*, std::array<double, 3>> point_params;

    for (auto match : frame.map_matches()) {
        const auto& point = match.point;
        if (point.observations().size() < MIN_OBSERVATIONS_TO_OPTIMIZE) {
            continue;
        }
        point_params.emplace(&point,
                             std::array<double, 3>{point.position().x(), point.position().y(), point.position().z()});
    }
    size_t reprojections = 0;
    for (const auto& match : frame.map_matches()) {
        if (point_params.find(&match.point) == point_params.end()) {
            continue;
        }
        double* point = point_params[&match.point].data();
        problem.AddResidualBlock(reprojection(frame, match.keypoint_index, camera),
                                 new ceres::HuberLoss(sqrt(5.991)),
                                 pose_params.data(),
                                 point);
        problem.SetParameterBlockConstant(point);
        reprojections++;
    }

    // Nothing to constrain
    if (reprojections == 0) {
        return false;
    }

    const auto* rotation_prior = std::get_if<RotationPrior>(&inertial);
    const bool rotation_valid = rotation_prior != nullptr && rotation_prior->enabled();

    const auto* imu_delta = std::get_if<InertialDelta>(&inertial);
    const bool imu_valid = imu_delta != nullptr && imu_delta->enabled();

    if (imu_valid) {
        previous_pose = pack_pose(*imu_delta->previous);
        previous_velocity = pack_velocity(*imu_delta->previous);
        previous_bias = pack_bias(*imu_delta->previous);
        velocity_params = pack_velocity(frame);
        problem.AddResidualBlock(imu_preintegration(imu_delta->summary, imu_delta->gravity),
                                 nullptr,
                                 previous_pose.data(),
                                 previous_velocity.data(),
                                 previous_bias.data(),
                                 pose_params.data(),
                                 velocity_params.data());
        for (double* block : {previous_pose.data(), previous_velocity.data(), previous_bias.data()}) {
            problem.SetParameterBlockConstant(block);
        }
    } else if (rotation_valid) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PredictedRotationError, 3, 6>(
                new PredictedRotationError(rotation_prior->predicted, rotation_prior->sigma_radians)),
            nullptr,
            pose_params.data());
    }
    if (!solve(problem, POSE_ITERATIONS, ceres::DENSE_QR)) {
        return false;
    }
    unpack_pose(pose_params, frame);
    if (imu_valid) {
        unpack_inertial(velocity_params, previous_bias, frame);
    }
    return true;
}

bool bundle_adjust(const std::vector<FrameConfig>& frames,
                   const Camera& camera,
                   Map& map,
                   const InertialInput& inertial)
{
    auto problem = ceres::Problem();
    std::unordered_map<const Frame*, std::array<double, 6>> frame_params;
    std::unordered_map<const Frame*, std::array<double, 3>> velocity_params;
    std::unordered_map<const Frame*, std::array<double, 6>> bias_params;
    std::unordered_map<const MapPoint*, std::array<double, 3>> map_point_params;

    // Add pose parameters
    for (const auto& frame_config : frames) {
        frame_params.emplace(frame_config.frame, pack_pose(*frame_config.frame));
        velocity_params.emplace(frame_config.frame, pack_velocity(*frame_config.frame));
        bias_params.emplace(frame_config.frame, pack_bias(*frame_config.frame));
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

    if (inertial.usable()) {
        for (size_t i = 0; i + 1 < frames.size(); i++) {
            Frame* prev_frame = frames[i].frame;
            Frame* next_frame = frames[i + 1].frame;
            // Only need IMU between frames being optimized
            if (frames_to_optimize.count(prev_frame) == 0 || frames_to_optimize.count(next_frame) == 0) {
                continue;
            }
            const double from = inertial.time_of(prev_frame->index());
            const double to = inertial.time_of(next_frame->index());

            const std::vector<imu::Sample> samples = inertial.stream->between(from, to);
            if (samples.size() < 2) {
                continue;
            }
            const imu::Preintegrated summary = imu::preintegrate(samples, inertial.noise, prev_frame->inertial().bias);

            problem.AddResidualBlock(imu_preintegration(summary, inertial.gravity),
                                     nullptr,
                                     frame_params[prev_frame].data(),
                                     velocity_params[prev_frame].data(),
                                     bias_params[prev_frame].data(),
                                     frame_params[next_frame].data(),
                                     velocity_params[next_frame].data());
            problem.AddResidualBlock(imu_bias_random_walk(summary.duration, inertial.noise),
                                     nullptr,
                                     bias_params[prev_frame].data(),
                                     bias_params[next_frame].data());
        }
    }

    for (auto& [frame, params] : frame_params) {
        const bool fixed = frames_to_optimize.find(frame) == frames_to_optimize.end();
        if (fixed && problem.HasParameterBlock(params.data())) {
            problem.SetParameterBlockConstant(params.data());
        }
        for (double* block : {velocity_params[frame].data(), bias_params[frame].data()}) {
            if (fixed && problem.HasParameterBlock(block)) {
                problem.SetParameterBlockConstant(block);
            }
        }
    }

    if (!solve(problem, BA_ITERATIONS, ceres::SPARSE_SCHUR)) {
        return false;
    }
    for (const auto& frame_config : frames) {
        if (frames_to_optimize.find(frame_config.frame) != frames_to_optimize.end()) {
            unpack_pose(frame_params[frame_config.frame], *frame_config.frame);
            unpack_inertial(velocity_params[frame_config.frame], bias_params[frame_config.frame], *frame_config.frame);
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

namespace {

constexpr double SEQ_SIGMA_ROT = 0.02;
constexpr double SEQ_SIGMA_TRANS = 0.2;
constexpr double LOOP_SIGMA_ROT = 0.05;
constexpr double LOOP_SIGMA_TRANS = 0.5;

class RelativePoseError {
  public:
    RelativePoseError(const Eigen::Matrix3d& R_meas,
                      const Eigen::Vector3d& t_meas,
                      double sigma_rot,
                      double sigma_trans)
        : m_R_meas(R_meas), m_t_meas(t_meas), m_sigma_rot(sigma_rot), m_sigma_trans(sigma_trans)
    {
    }

    template <typename T> bool operator()(const T* const pose_from, const T* const pose_to, T* residuals) const
    {
        T R_from_data[9];
        T R_to_data[9];
        ceres::AngleAxisToRotationMatrix(pose_from, R_from_data);
        ceres::AngleAxisToRotationMatrix(pose_to, R_to_data);
        const Eigen::Map<const Eigen::Matrix<T, 3, 3>> R_from(R_from_data);
        const Eigen::Map<const Eigen::Matrix<T, 3, 3>> R_to(R_to_data);
        const Eigen::Matrix<T, 3, 1> c_from(pose_from[3], pose_from[4], pose_from[5]);
        const Eigen::Matrix<T, 3, 1> c_to(pose_to[3], pose_to[4], pose_to[5]);

        const Eigen::Matrix<T, 3, 3> R_est = R_from * R_to.transpose();
        const Eigen::Matrix<T, 3, 1> t_est = R_from * (c_to - c_from);
        const Eigen::Matrix<T, 3, 3> R_err = m_R_meas.transpose().cast<T>() * R_est;
        T rvec[3];
        ceres::RotationMatrixToAngleAxis(R_err.data(), rvec);
        residuals[0] = rvec[0] / T(m_sigma_rot);
        residuals[1] = rvec[1] / T(m_sigma_rot);
        residuals[2] = rvec[2] / T(m_sigma_rot);
        residuals[3] = (t_est[0] - T(m_t_meas[0])) / T(m_sigma_trans);
        residuals[4] = (t_est[1] - T(m_t_meas[1])) / T(m_sigma_trans);
        residuals[5] = (t_est[2] - T(m_t_meas[2])) / T(m_sigma_trans);
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Matrix4d& relative, double sigma_rot, double sigma_trans)
    {
        return new ceres::AutoDiffCostFunction<RelativePoseError, 6, 6, 6>(
            new RelativePoseError(relative.block<3, 3>(0, 0), relative.block<3, 1>(0, 3), sigma_rot, sigma_trans));
    }

  private:
    const Eigen::Matrix3d m_R_meas;
    const Eigen::Vector3d m_t_meas;
    const double m_sigma_rot;
    const double m_sigma_trans;
};

// xyz + yaw, since IMU gravity constrains pitch and roll
class RelativePose4DoFError {
  public:
    RelativePose4DoFError(const Eigen::Matrix3d& R_from0,
                          const Eigen::Matrix3d& R_to0,
                          const Eigen::Vector3d& up,
                          const Eigen::Matrix3d& R_meas,
                          const Eigen::Vector3d& t_meas,
                          double sigma_rot,
                          double sigma_trans)
        : m_R_from0(R_from0), m_R_to0(R_to0), m_up(up), m_R_meas(R_meas), m_t_meas(t_meas), m_sigma_rot(sigma_rot),
          m_sigma_trans(sigma_trans)
    {
    }

    template <typename T> Eigen::Matrix<T, 3, 3> rotation_cw(const T yaw, const Eigen::Matrix3d& R0) const
    {
        T aa[3] = {T(-m_up[0]) * yaw, T(-m_up[1]) * yaw, T(-m_up[2]) * yaw};
        T R_delta[9];
        ceres::AngleAxisToRotationMatrix(aa, R_delta);
        return R0.cast<T>() * Eigen::Map<const Eigen::Matrix<T, 3, 3>>(R_delta);
    }

    template <typename T> bool operator()(const T* const pose_from, const T* const pose_to, T* residuals) const
    {
        const Eigen::Matrix<T, 3, 3> R_from = rotation_cw(pose_from[0], m_R_from0);
        const Eigen::Matrix<T, 3, 3> R_to = rotation_cw(pose_to[0], m_R_to0);
        const Eigen::Matrix<T, 3, 1> c_from(pose_from[1], pose_from[2], pose_from[3]);
        const Eigen::Matrix<T, 3, 1> c_to(pose_to[1], pose_to[2], pose_to[3]);

        const Eigen::Matrix<T, 3, 3> R_est = R_from * R_to.transpose();
        const Eigen::Matrix<T, 3, 1> t_est = R_from * (c_to - c_from);
        const Eigen::Matrix<T, 3, 3> R_err = m_R_meas.transpose().cast<T>() * R_est;
        T rvec[3];
        ceres::RotationMatrixToAngleAxis(R_err.data(), rvec);
        residuals[0] = rvec[0] / T(m_sigma_rot);
        residuals[1] = rvec[1] / T(m_sigma_rot);
        residuals[2] = rvec[2] / T(m_sigma_rot);
        residuals[3] = (t_est[0] - T(m_t_meas[0])) / T(m_sigma_trans);
        residuals[4] = (t_est[1] - T(m_t_meas[1])) / T(m_sigma_trans);
        residuals[5] = (t_est[2] - T(m_t_meas[2])) / T(m_sigma_trans);
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Matrix3d& R_from0,
                                       const Eigen::Matrix3d& R_to0,
                                       const Eigen::Vector3d& up,
                                       const Eigen::Matrix4d& relative,
                                       double sigma_rot,
                                       double sigma_trans)
    {
        return new ceres::AutoDiffCostFunction<RelativePose4DoFError, 6, 4, 4>(new RelativePose4DoFError(
            R_from0, R_to0, up, relative.block<3, 3>(0, 0), relative.block<3, 1>(0, 3), sigma_rot, sigma_trans));
    }

  private:
    const Eigen::Matrix3d m_R_from0;
    const Eigen::Matrix3d m_R_to0;
    const Eigen::Vector3d m_up;
    const Eigen::Matrix3d m_R_meas;
    const Eigen::Vector3d m_t_meas;
    const double m_sigma_rot;
    const double m_sigma_trans;
};

Eigen::Matrix4d pose_relative(const Frame& from, const Frame& to)
{
    return from.pose().cast<double>() * to.pose().inverse().cast<double>();
}

void apply_corrected_pose(Frame& frame, const Eigen::Matrix3d& R_cw, const Eigen::Vector3d& center)
{
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    pose.block<3, 3>(0, 0) = R_cw.cast<float>();
    pose.block<3, 1>(0, 3) = -R_cw.cast<float>() * center.cast<float>();
    const Eigen::Matrix3f R_delta = pose.block<3, 3>(0, 0).transpose() * frame.pose().block<3, 3>(0, 0);
    frame.set_pose(pose);
    InertialState inertial = frame.inertial();
    inertial.velocity = R_delta.cast<double>() * inertial.velocity;
    frame.set_inertial(inertial);
}

void transform_points(Map& map, const std::unordered_map<const Frame*, Eigen::Matrix4f>& old_poses)
{
    for (auto& point : map) {
        if (point.observations().empty()) {
            continue;
        }
        KeyFrame* owner = nullptr;
        for (const auto& [observer, _] : point.observations()) {
            if (owner == nullptr || observer->index() < owner->index()) {
                owner = observer;
            }
        }
        const auto old = old_poses.find(owner);
        if (old == old_poses.end()) {
            continue;
        }
        const Eigen::Matrix4f& T_old = old->second;
        const Eigen::Matrix4f& T_new = owner->pose();
        const Eigen::Vector3f p = point.position();
        const Eigen::Vector3f p_cam = T_old.block<3, 3>(0, 0) * p + T_old.block<3, 1>(0, 3);
        const Eigen::Vector3f p_new = T_new.block<3, 3>(0, 0).transpose() * (p_cam - T_new.block<3, 1>(0, 3));
        point.set_position(p_new);
    }
}

} // namespace

bool pose_graph(const std::vector<std::shared_ptr<KeyFrame>>& key_frames,
                const std::vector<PoseGraphConstraint>& loops,
                Map& map,
                bool four_dof,
                const Eigen::Vector3d& gravity)
{
    if (key_frames.size() < 3 || loops.empty()) {
        return false;
    }

    Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
    if (four_dof) {
        if (gravity.squaredNorm() < 1e-6) {
            four_dof = false;
        } else {
            up = -gravity.normalized();
        }
    }

    auto problem = ceres::Problem();
    std::vector<std::array<double, 6>> se3(key_frames.size());
    std::vector<std::array<double, 4>> dof4(key_frames.size());
    std::vector<Eigen::Matrix3d> R0(key_frames.size());
    std::unordered_map<const Frame*, Eigen::Matrix4f> old_poses;
    old_poses.reserve(key_frames.size());

    for (size_t i = 0; i < key_frames.size(); i++) {
        old_poses[key_frames[i].get()] = key_frames[i]->pose();
        se3[i] = pack_pose(*key_frames[i]);
        R0[i] = key_frames[i]->pose().block<3, 3>(0, 0).cast<double>();
        const Eigen::Vector3f center = key_frames[i]->camera_center();
        dof4[i] = {0.0, center.x(), center.y(), center.z()};
    }

    auto add_edge = [&](size_t from, size_t to, const Eigen::Matrix4d& relative, bool loop) {
        const double sigma_rot = loop ? LOOP_SIGMA_ROT : SEQ_SIGMA_ROT;
        const double sigma_trans = loop ? LOOP_SIGMA_TRANS : SEQ_SIGMA_TRANS;
        ceres::LossFunction* loss = loop ? new ceres::HuberLoss(1.0) : nullptr;
        if (four_dof) {
            problem.AddResidualBlock(
                RelativePose4DoFError::Create(R0[from], R0[to], up, relative, sigma_rot, sigma_trans),
                loss,
                dof4[from].data(),
                dof4[to].data());
        } else {
            problem.AddResidualBlock(
                RelativePoseError::Create(relative, sigma_rot, sigma_trans), loss, se3[from].data(), se3[to].data());
        }
    };

    for (size_t i = 0; i + 1 < key_frames.size(); i++) {
        add_edge(i, i + 1, pose_relative(*key_frames[i], *key_frames[i + 1]), false);
    }
    for (const auto& loop : loops) {
        if (loop.from >= key_frames.size() || loop.to >= key_frames.size() || loop.from == loop.to) {
            continue;
        }
        add_edge(loop.from, loop.to, loop.relative, true);
    }

    if (four_dof) {
        problem.SetParameterBlockConstant(dof4[0].data());
    } else {
        problem.SetParameterBlockConstant(se3[0].data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = PGO_ITERATIONS;
    options.num_threads = solver_threads();
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << "Pose graph " << (four_dof ? "4-DOF" : "SE3") << " " << summary.BriefReport() << '\n';
    const bool usable =
        summary.IsSolutionUsable() && std::isfinite(summary.final_cost) && summary.final_cost <= summary.initial_cost;
    if (!usable) {
        std::cout << "Pose graph rejected, unusable or non-improving solution\n";
        return false;
    }

    float max_correction = 0.0F;
    for (size_t i = 0; i < key_frames.size(); i++) {
        Eigen::Matrix3d R_cw;
        Eigen::Vector3d center;
        if (four_dof) {
            const double yaw = dof4[i][0];
            R_cw = R0[i] * Eigen::AngleAxisd(-yaw, up).toRotationMatrix();
            center = Eigen::Vector3d(dof4[i][1], dof4[i][2], dof4[i][3]);
        } else {
            R_cw = rodrigues_to_matrix(Eigen::Vector3f(se3[i][0], se3[i][1], se3[i][2])).cast<double>();
            center = Eigen::Vector3d(se3[i][3], se3[i][4], se3[i][5]);
        }
        const Eigen::Vector3f before = key_frames[i]->camera_center();
        apply_corrected_pose(*key_frames[i], R_cw, center);
        max_correction = std::max(max_correction, (key_frames[i]->camera_center() - before).norm());
    }
    transform_points(map, old_poses);
    std::cout << "Pose graph applied, max snap " << max_correction << " loops " << loops.size() << '\n';
    return true;
}

} // namespace slam::optimization
