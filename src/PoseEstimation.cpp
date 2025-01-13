#include "PoseEstimation.h"

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include "Features.h"
#include "Triangulation.h"

namespace slam::pose {

static Eigen::Matrix4f pose_from_Rt(const cv::Mat& R, const cv::Mat& t)
{
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            pose(i, j) = R.at<double>(i, j);
        }
        pose(i, 3) = t.at<double>(i, 0);
    }
    return pose;
}

static Eigen::Matrix4f recover_pose_from_essential(const cv::Mat& E,
                                                   const Camera& camera,
                                                   const std::vector<cv::Point2f>& points_from,
                                                   const std::vector<cv::Point2f>& points_to,
                                                   const std::vector<u_char>& inliers)
{
    cv::Mat R1, R2, t;
    cv::decomposeEssentialMat(E, R1, R2, t);
    std::vector<Eigen::Matrix4f> poses = {
        pose_from_Rt(R1, t),
        pose_from_Rt(R1, -t),
        pose_from_Rt(R2, t),
        pose_from_Rt(R2, -t),
    };

    std::vector<Eigen::Vector2f> points_from_eigen;
    std::vector<Eigen::Vector2f> points_to_eigen;
    for (int i = 0; i < points_from.size(); i++) {
        points_from_eigen.push_back(Eigen::Vector2f(points_from[i].x, points_from[i].y));
        points_to_eigen.push_back(Eigen::Vector2f(points_to[i].x, points_to[i].y));
    }

    int best_pose_index = 0;
    int most_visible_points = 0;
    for (int i = 0; i < poses.size(); i++) {
        auto triangulated_points = triangulation::triangulate_points(points_from_eigen,
                                                                     points_to_eigen,
                                                                     Eigen::Matrix4f::Identity(),
                                                                     poses[i],
                                                                     camera);
        if (triangulated_points.size() > most_visible_points) {
            most_visible_points = triangulated_points.size();
            best_pose_index = i;
        }
    }

    return poses[best_pose_index];
}

PoseEstimate estimate_pose(const ExtractedFeatures& prev_features,
                           const ExtractedFeatures& features,
                           const std::vector<FeatureMatch>& matches,
                           const Camera& camera)
{
    std::vector<cv::Point2f> matched_points_from, matched_points_to;
    for (const auto& match : matches) {
        matched_points_from.push_back(prev_features.keypoints[match.train_index].pt);
        matched_points_to.push_back(features.keypoints[match.query_index].pt);
    }

    std::vector<uchar> essential_inliers;
    cv::Mat E = cv::findEssentialMat(matched_points_from,
                                     matched_points_to,
                                     cv_utils::intrinsic_mat_cv(camera),
                                     cv::RANSAC,
                                     0.999,
                                     0.4,
                                     essential_inliers);

    PoseEstimate pose_estimate;
    pose_estimate.pose = recover_pose_from_essential(E,
                                                     camera,
                                                     matched_points_from,
                                                     matched_points_to,
                                                     essential_inliers);
    for (int i = 0; i < matches.size(); i++) {
        if (essential_inliers[i]) {
            pose_estimate.inlier_matches.push_back(matches[i]);
        }
    }
    return pose_estimate;
}

} // namespace slam::pose