#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core.hpp>

#include "BundleAdjustment.h"

namespace slam {

class ReprojectionError {
public:
    ReprojectionError(const cv::Point2f& observation, double focal_length, const cv::Point2f& principal_point)
        : m_observation(observation), m_focal_length(focal_length), m_principal_point(principal_point) {}

    // The camera params (angle/translation) should transform from world coords to camera coords
    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const {
        const T* camera_rotation = &camera[0];
        const T* camera_translation = &camera[3];

        T p[3];
        ceres::AngleAxisRotatePoint(camera_rotation, point, p);
        p[0] += camera_translation[0];
        p[1] += camera_translation[1];
        p[2] += camera_translation[2];

        T normalized_obs_x = (T(m_observation.x) - T(m_principal_point.x)) / T(m_focal_length);
        T normalized_obs_y = (T(m_observation.y) - T(m_principal_point.y)) / T(m_focal_length);

        residuals[0] = p[0] / p[2] - normalized_obs_x;
        residuals[1] = p[1] / p[2] - normalized_obs_y;

        return true;
    }

private:
    const cv::Point2f m_observation;
    const double m_focal_length;
    const cv::Point2f m_principal_point;
};

static cv::Mat rodrigues_to_matrix(const cv::Vec3d& rvec) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    return R;
}

static cv::Vec3d matrix_to_rodrigues(const cv::Mat& R) {
    cv::Vec3d rvec;
    cv::Rodrigues(R, rvec);
    return rvec;
}

static bool frameInWindow(Map& map, Frame* frame, int window) {
    if (window < 1) return true;
    return frame->get_id() >= (map.get_frames().size() - window);
}

void bundle_adjustment(Map& map, int window, bool optimize_points) {
    auto problem = std::make_unique<ceres::Problem>();
    
    std::unordered_map<int, std::array<double, 6>> camera_poses;  // [angle-axis (3), translation (3)]
    std::unordered_map<int, std::array<double, 3>> points;
    
    double focal_length = map.get_camera().get_intrinsic_matrix().at<double>(0, 0);
    cv::Point2f principal_point(
        map.get_camera().get_width() / 2.0f, 
        map.get_camera().get_height() / 2.0f
    );

    // Add poses
    for (const auto& frame : map.get_frames()) {
        if (!optimize_points && !frameInWindow(map, frame.get(), window)) continue;

        cv::Mat pose_inv = frame->get_pose().inv();
        cv::Mat R = pose_inv(cv::Rect(0, 0, 3, 3));
        cv::Mat t = pose_inv(cv::Rect(3, 0, 1, 3));
        
        cv::Vec3d rvec = matrix_to_rodrigues(R);

        // Add an entry for all points we want included
        for (int i = 0; i < frame->get_keypoints().size(); i++) {
            MapPoint* map_point = frame->get_corresponding_map_point(i);
            if (!map_point) continue;

            auto& pose_params = camera_poses[frame->get_id()];
            
            // Rotation
            pose_params[0] = rvec[0];
            pose_params[1] = rvec[1];
            pose_params[2] = rvec[2];
            
            // Translation
            pose_params[3] = t.at<double>(0);
            pose_params[4] = t.at<double>(1);
            pose_params[5] = t.at<double>(2);

            // Add point
            auto& point = points[map_point->get_id()];
            cv::Point3f pos = map_point->get_position();
            point[0] = pos.x;
            point[1] = pos.y;
            point[2] = pos.z;

            // Add edge
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
                    new ReprojectionError(frame->get_keypoint(i).pt, focal_length, principal_point));
            problem->AddResidualBlock(
                cost_function,
                new ceres::HuberLoss(sqrt(5.991)),
                pose_params.data(),
                point.data()
            );

            if (frame->get_id() < 2 || !frameInWindow(map, frame.get(), window)) {
                problem->SetParameterBlockConstant(pose_params.data());
            }
        }
    }

    // Solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 10;

    ceres::Solver::Summary summary;
    ceres::Solve(options, problem.get(), &summary);
    std::cout << summary.BriefReport() << std::endl;

    // Update poses and points
    for (const auto& frame : map.get_frames()) {
        if (camera_poses.find(frame->get_id()) == camera_poses.end()) continue;

        auto& pose_params = camera_poses[frame->get_id()];
        if (problem->IsParameterBlockConstant(pose_params.data())) continue;
        
        cv::Vec3d rvec(pose_params[0], pose_params[1], pose_params[2]);
        cv::Mat R = rodrigues_to_matrix(rvec);
        
        cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
        R.copyTo(pose(cv::Rect(0, 0, 3, 3)));
        pose.at<double>(0, 3) = pose_params[3];
        pose.at<double>(1, 3) = pose_params[4];
        pose.at<double>(2, 3) = pose_params[5];
        
        frame->set_pose(pose.inv());
    }

    if (optimize_points) {
        for (const auto& [id, point] : points) {
            cv::Point3f pos(point[0], point[1], point[2]);
            map.get_map_point(id).set_position(pos);
        }
    }
}

};