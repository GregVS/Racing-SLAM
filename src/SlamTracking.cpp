#include "Slam.h"

#include "Frame.h"
#include "MotionModel.h"
#include "Optimization.h"
#include "PoseEstimation.h"
#include "YawEstimation.h"
#include "features/FeatureExtractor.h"

namespace slam {

namespace {

constexpr int KLT_WINDOW = 21;
constexpr int KLT_PYRAMID_LEVELS = 4;
constexpr float KLT_MAX_FORWARD_BACKWARD_ERROR = 1.0F;
constexpr int KLT_REPLENISH_RADIUS = 5;
constexpr size_t MAX_TRACKED_FEATURES = 2000;
constexpr size_t MIN_PNP_POINTS = 15;
constexpr float PNP_REPROJECTION_ERROR = 4.0F;
constexpr float MIN_PNP_INLIER_RATIO = 0.5F;

cv::Mat to_gray(const cv::Mat& image)
{
    if (image.channels() == 1) {
        return image;
    }
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

} // namespace

std::pair<ExtractedFeatures, std::vector<FeatureMatch>> Slam::track_features(const cv::Mat& image)
{
    cv::Mat prev_gray = to_gray(m_last_frame->image());
    cv::Mat next_gray = to_gray(image);

    const auto& prev_features = m_last_frame->features();
    std::vector<cv::Point2f> prev_points;
    prev_points.reserve(prev_features.keypoints.size());
    for (const auto& keypoint : prev_features.keypoints) {
        prev_points.push_back(keypoint.pt);
    }

    std::vector<cv::Point2f> next_points;
    std::vector<cv::Point2f> back_points;
    std::vector<uchar> forward_ok;
    std::vector<uchar> backward_ok;
    auto window = cv::Size(KLT_WINDOW, KLT_WINDOW);
    cv::calcOpticalFlowPyrLK(
        prev_gray, next_gray, prev_points, next_points, forward_ok, cv::noArray(), window, KLT_PYRAMID_LEVELS);
    cv::calcOpticalFlowPyrLK(
        next_gray, prev_gray, next_points, back_points, backward_ok, cv::noArray(), window, KLT_PYRAMID_LEVELS);

    ExtractedFeatures features;
    std::vector<FeatureMatch> matches;
    features.keypoints.reserve(prev_points.size());
    matches.reserve(prev_points.size());
    cv::Mat replenish_mask = m_static_mask.clone();
    for (size_t i = 0; i < prev_points.size(); i++) {
        if (!forward_ok[i] || !backward_ok[i] ||
            cv::norm(prev_points[i] - back_points[i]) > KLT_MAX_FORWARD_BACKWARD_ERROR) {
            continue;
        }
        auto point = cv::Point(cvRound(next_points[i].x), cvRound(next_points[i].y));
        if (point.x < 0 || point.y < 0 || point.x >= next_gray.cols || point.y >= next_gray.rows ||
            m_static_mask.at<uchar>(point) == 0) {
            continue;
        }

        auto keypoint = prev_features.keypoints[i];
        keypoint.pt = next_points[i];
        matches.emplace_back(static_cast<int>(i), static_cast<int>(features.keypoints.size()));
        features.keypoints.push_back(keypoint);
        features.descriptors.push_back(prev_features.descriptors.row(i));
        cv::circle(replenish_mask, point, KLT_REPLENISH_RADIUS, 0, -1);
    }

    // Refill where nothing is tracked so coverage does not decay as points leave the image;
    // the detector returns corners strongest first, so capping the total keeps the best
    auto new_features = m_feature_extractor->extract_features(image, replenish_mask);

    size_t budget =
        features.keypoints.size() < MAX_TRACKED_FEATURES ? MAX_TRACKED_FEATURES - features.keypoints.size() : 0;
    for (size_t i = 0; i < new_features.keypoints.size() && budget > 0; i++) {
        features.keypoints.push_back(new_features.keypoints[i]);
        features.descriptors.push_back(new_features.descriptors.row(i));
        budget--;
    }

    // Re-describe against this frame so descriptors follow the viewpoint instead of staying
    // frozen at first detection
    features.descriptors = m_feature_extractor->refresh_descriptors(image, features);

    std::cout << "Tracked features: " << matches.size() << " of " << prev_points.size() << ", replenished "
              << new_features.keypoints.size() << '\n';
    return {features, matches};
}

std::vector<FeatureMatch> Slam::initial_pose_estimate(Frame& frame, const std::vector<FeatureMatch>& matches)
{
    if (m_config.essential_matrix_estimation || m_key_frames.size() < 2) {
        auto pose_estimate = pose::estimate_pose(m_last_frame->features(), frame.features(), matches, m_camera);
        // Wheel odometry owns distance. Vision supplies orientation while the nonholonomic
        // camera-forward model supplies translation direction.
        auto index = m_last_frame->index();
        if (index >= 1 && index < m_trajectory.size()) {
            float last_step =
                (motion::camera_center(m_trajectory[index]) - motion::camera_center(m_trajectory[index - 1])).norm();
            if (!m_config.metric_steps.empty()) {
                last_step = metric_distance(index, frame.index());
            }
            Eigen::Matrix4f relative = pose_estimate.pose;
            if (!m_config.metric_steps.empty()) {
                float max_yaw =
                    motion::MAX_ANGULAR_SPEED_DEGREES * m_config.seconds_per_frame * 3.14159265358979323846F / 180.0F;
                auto yaw_estimate =
                    yaw::estimate(m_last_frame->features(), frame.features(), matches, m_camera, max_yaw);
                Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
                bool essential_healthy = motion::is_rotation_plausible(identity, relative, m_config.seconds_per_frame);
                if (!essential_healthy && yaw_estimate.healthy) {
                    relative = Eigen::Matrix4f::Identity();
                    relative.block<3, 3>(0, 0) =
                        Eigen::AngleAxisf(yaw_estimate.radians, Eigen::Vector3f::UnitY()).toRotationMatrix();
                    std::cout << "Unhealthy essential rotation replaced by bounded yaw\n";
                }
            }
            Eigen::Matrix4f candidate = relative * m_last_frame->pose();
            if (!m_config.metric_steps.empty() && last_step > 1e-6F) {
                if (!motion::is_rotation_plausible(m_last_frame->pose(), candidate, m_config.seconds_per_frame)) {
                    std::cout << "Essential rotation rejected by temporal motion bound\n";
                    candidate = m_last_frame->pose();
                }
                frame.set_pose(motion::with_metric_step(m_last_frame->pose(), candidate, last_step));
                return pose_estimate.inlier_matches;
            }

            auto scaled = pose_estimate.pose;
            if (scaled.block<3, 1>(0, 3).norm() > 1e-6F && last_step > 1e-6F) {
                scaled.block<3, 1>(0, 3) *= last_step / scaled.block<3, 1>(0, 3).norm();
                frame.set_pose(scaled * m_last_frame->pose());
                return pose_estimate.inlier_matches;
            }
        }
        frame.set_pose(pose_estimate.pose * m_last_frame->pose());
        return pose_estimate.inlier_matches;
    }
    frame.set_pose(m_last_frame->pose());
    return {};
}

void Slam::track_from_last_frame(Frame& frame, const std::vector<FeatureMatch>& matches)
{
    // The previous frame's map points carry across the 2D-2D matches without needing a pose,
    // so this is the one place the translation magnitude can be recovered
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    std::vector<const MapPoint*> points;
    std::vector<size_t> keypoint_indices;
    object_points.reserve(matches.size());
    image_points.reserve(matches.size());
    points.reserve(matches.size());
    keypoint_indices.reserve(matches.size());
    for (const auto& match : matches) {
        if (!m_last_frame->is_matched(match.train_index)) {
            continue;
        }
        const auto& point = m_last_frame->map_match(match.train_index);
        if (point.observations().size() < 2 && !point.track_consistent()) {
            continue;
        }
        object_points.emplace_back(point.position().x(), point.position().y(), point.position().z());
        image_points.push_back(frame.keypoint(match.query_index).pt);
        points.push_back(&point);
        keypoint_indices.push_back(match.query_index);
    }

    if (object_points.size() < MIN_PNP_POINTS) {
        std::cout << "Too few correspondences to track from last frame\n";
        return;
    }

    // RANSAC rather than least squares: a robust kernel only down-weights wrong matches where
    // consensus excludes them outright
    cv::Mat intrinsics;
    cv_utils::intrinsic_mat_cv(m_camera).convertTo(intrinsics, CV_64F);
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat inliers;
    bool solved = cv::solvePnPRansac(object_points,
                                     image_points,
                                     intrinsics,
                                     cv::Mat(),
                                     rvec,
                                     tvec,
                                     false,
                                     200,
                                     PNP_REPROJECTION_ERROR,
                                     0.99,
                                     inliers,
                                     cv::SOLVEPNP_EPNP);
    float inlier_ratio = solved ? static_cast<float>(inliers.rows) / static_cast<float>(object_points.size()) : 0.0F;
    if (!solved || inliers.rows < static_cast<int>(MIN_PNP_POINTS) || inlier_ratio < MIN_PNP_INLIER_RATIO) {
        std::cout << "RANSAC pose rejected, keeping essential matrix estimate\n";
        return;
    }

    cv::Mat rotation;
    cv::Rodrigues(rvec, rotation);
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            pose(i, j) = rotation.at<double>(i, j);
        }
        pose(i, 3) = tvec.at<double>(i, 0);
    }

    if (!motion::is_rotation_plausible(m_last_frame->pose(), pose, m_config.seconds_per_frame)) {
        std::cout << "RANSAC pose rejected by temporal motion bound\n";
        return;
    }
    if (m_config.metric_steps.empty()) {
        frame.set_pose(pose);
    }

    // PnP is a correspondence-health check only when wheel odometry supplies metric motion.
    // Its unconstrained pose is too fragile on planar racing scenes to own the vehicle state.
    for (int i = 0; i < inliers.rows; i++) {
        int index = inliers.at<int>(i, 0);
        if (frame.is_matched(keypoint_indices[index]) || frame.is_matched(*points[index])) {
            continue;
        }
        frame.add_map_match(MapPointMatch{*points[index], keypoint_indices[index]});
    }
    std::cout << "Tracked from last frame: " << inliers.rows << " / " << object_points.size() << '\n';
}

void Slam::update_tracks(const std::vector<FeatureMatch>& matches)
{
    // train_index indexes the previous frame, query_index the current one
    std::unordered_map<size_t, FeatureTrack> carried;
    carried.reserve(matches.size());
    for (const auto& match : matches) {
        auto existing = m_tracks.find(match.train_index);
        carried[match.query_index] = existing != m_tracks.end() ? existing->second : FeatureTrack{};
    }
    m_tracks = std::move(carried);
}

void Slam::match_with_last_key_frame(Frame& frame)
{
    const auto& last_frame = m_key_frames.back();
    auto map_matches = m_feature_extractor->match_features(
        frame, m_camera, m_map, [&](const MapPoint& point) { return point.is_observed_by(last_frame.get()); });
    for (const auto& match : map_matches) {
        frame.add_map_match(match);
    }
    std::cout << "Map matches with last frame: " << map_matches.size() << '\n';
}

void Slam::match_with_map(Frame& frame)
{
    auto map_matches =
        m_feature_extractor->match_features(frame, m_camera, m_map, [&](const MapPoint& point) { return true; });
    for (const auto& match : map_matches) {
        frame.add_map_match(match);
    }
    std::cout << "Number of map matches: " << map_matches.size() << '\n';
}

void Slam::optimize_pose(Frame& frame)
{
    if (!m_config.optimize_pose || !m_config.metric_steps.empty()) {
        return;
    }
    if (frame.num_map_matches() < MIN_PNP_POINTS) {
        return;
    }

    // Motion-only BA
    auto original_pose = frame.pose();
    auto config = optimization::OptimizationConfig{
        .optimize_points = false,
        .frames = {{true, &frame, m_config.metric_steps.empty()}},
    };
    bool optimized = optimization::optimize(config, m_camera, m_map);
    if (!optimized || !motion::is_rotation_plausible(m_last_frame->pose(), frame.pose(), m_config.seconds_per_frame)) {
        frame.set_pose(original_pose);
        if (optimized) {
            std::cout << "Pose optimization rolled back by temporal motion bound\n";
        }
    }
}

} // namespace slam
