#include "PoseEstimation.h"

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <random>

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
        auto triangulated_points = triangulation::triangulate_points(
            points_from_eigen, points_to_eigen, Eigen::Matrix4f::Identity(), poses[i], camera);
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
                                     cv::USAC_ACCURATE,
                                     0.99,
                                     1.0,
                                     essential_inliers);

    PoseEstimate pose_estimate;
    pose_estimate.pose =
        recover_pose_from_essential(E, camera, matched_points_from, matched_points_to, essential_inliers);
    for (int i = 0; i < matches.size(); i++) {
        if (essential_inliers[i]) {
            pose_estimate.inlier_matches.push_back(matches[i]);
        }
    }
    return pose_estimate;
}

namespace {

/** Sampson distance in pixels */
float epipolar_error(const Eigen::Matrix3f& essential,
                     const Eigen::Vector3f& from,
                     const Eigen::Vector3f& to,
                     float focal)
{
    const Eigen::Vector3f line_to = essential * from;
    const Eigen::Vector3f line_from = essential.transpose() * to;
    const float denominator = line_to.head<2>().squaredNorm() + line_from.head<2>().squaredNorm();
    if (denominator < 1e-12F) {
        return std::numeric_limits<float>::max();
    }
    const float numerator = to.dot(line_to);
    return focal * std::abs(numerator) / std::sqrt(denominator);
}

} // namespace

PoseEstimate estimate_pose_with_known_rotation(const ExtractedFeatures& prev_features,
                                               const ExtractedFeatures& features,
                                               const std::vector<FeatureMatch>& matches,
                                               const Camera& camera,
                                               const Eigen::Matrix3f& rotation)
{
    constexpr float MAX_EPIPOLAR_ERROR = 2.0F;
    constexpr size_t ITERATIONS = 200;

    PoseEstimate estimate;
    estimate.pose = Eigen::Matrix4f::Identity();
    estimate.pose.block<3, 3>(0, 0) = rotation;
    if (matches.size() < 8) {
        return estimate;
    }

    const Eigen::Matrix3f inverse_intrinsics = camera.get_intrinsic_matrix().inverse();
    const float focal = camera.get_intrinsic_matrix()(0, 0);

    std::vector<Eigen::Vector3f> constraints;
    std::vector<Eigen::Vector3f> from_rays;
    std::vector<Eigen::Vector3f> to_rays;
    constraints.reserve(matches.size());
    for (const auto& match : matches) {
        const auto& a = prev_features.keypoints[match.train_index].pt;
        const auto& b = features.keypoints[match.query_index].pt;
        const Eigen::Vector3f from = inverse_intrinsics * Eigen::Vector3f(a.x, a.y, 1.0F);
        const Eigen::Vector3f to = inverse_intrinsics * Eigen::Vector3f(b.x, b.y, 1.0F);
        from_rays.push_back(from);
        to_rays.push_back(to);
        constraints.push_back((rotation * from).cross(to));
    }

    std::mt19937 generator(0);
    std::uniform_int_distribution<size_t> pick(0, constraints.size() - 1);
    Eigen::Vector3f best_translation = Eigen::Vector3f::UnitZ();
    size_t best_support = 0;

    for (size_t iteration = 0; iteration < ITERATIONS; iteration++) {
        const size_t i = pick(generator);
        const size_t j = pick(generator);
        if (i == j) {
            continue;
        }
        Eigen::Vector3f translation = constraints[i].cross(constraints[j]);
        if (translation.norm() < 1e-9F) {
            continue;
        }
        translation.normalize();

        Eigen::Matrix3f essential;
        essential << 0.0F, -translation.z(), translation.y(), translation.z(), 0.0F, -translation.x(), -translation.y(),
            translation.x(), 0.0F;
        essential = essential * rotation;

        size_t support = 0;
        for (size_t k = 0; k < constraints.size(); k++) {
            if (epipolar_error(essential, from_rays[k], to_rays[k], focal) < MAX_EPIPOLAR_ERROR) {
                support++;
            }
        }
        if (support > best_support) {
            best_support = support;
            best_translation = translation;
        }
    }
    if (best_support < 8) {
        return estimate;
    }

    // Refit on all inliers
    std::vector<size_t> inliers;
    Eigen::Matrix3f essential;
    essential << 0.0F, -best_translation.z(), best_translation.y(), best_translation.z(), 0.0F, -best_translation.x(),
        -best_translation.y(), best_translation.x(), 0.0F;
    essential = essential * rotation;
    for (size_t k = 0; k < constraints.size(); k++) {
        if (epipolar_error(essential, from_rays[k], to_rays[k], focal) < MAX_EPIPOLAR_ERROR) {
            inliers.push_back(k);
        }
    }

    Eigen::MatrixXf stack(inliers.size(), 3);
    for (size_t k = 0; k < inliers.size(); k++) {
        stack.row(k) = constraints[inliers[k]].transpose();
    }
    const Eigen::JacobiSVD<Eigen::MatrixXf> decomposition(stack, Eigen::ComputeThinV);
    Eigen::Vector3f translation = decomposition.matrixV().col(2);
    if (translation.dot(best_translation) < 0.0F) {
        translation = -translation;
    }

    // Choose the sign of the baseline that puts more points in front of both cameras
    const auto in_front = [&](const Eigen::Vector3f& candidate) {
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        pose.block<3, 3>(0, 0) = rotation;
        pose.block<3, 1>(0, 3) = candidate;
        std::vector<Eigen::Vector2f> from_points;
        std::vector<Eigen::Vector2f> to_points;
        for (size_t k : inliers) {
            const auto& a = prev_features.keypoints[matches[k].train_index].pt;
            const auto& b = features.keypoints[matches[k].query_index].pt;
            from_points.emplace_back(a.x, a.y);
            to_points.emplace_back(b.x, b.y);
        }
        return triangulation::triangulate_points(from_points, to_points, Eigen::Matrix4f::Identity(), pose, camera)
            .size();
    };
    if (in_front(-translation) > in_front(translation)) {
        translation = -translation;
    }

    estimate.pose.block<3, 1>(0, 3) = translation;
    for (size_t k : inliers) {
        estimate.inlier_matches.push_back(matches[k]);
    }
    return estimate;
}

} // namespace slam::pose
