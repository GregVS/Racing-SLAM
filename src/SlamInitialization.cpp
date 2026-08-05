#include "Slam.h"

#include <algorithm>

#include "Frame.h"
#include "Optimization.h"
#include "PoseEstimation.h"
#include "Triangulation.h"
#include "features/FeatureExtractor.h"

namespace slam {

namespace {

constexpr int KLT_WINDOW = 21;
constexpr int KLT_PYRAMID_LEVELS = 4;
constexpr float KLT_MAX_FORWARD_BACKWARD_ERROR = 1.0F;

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

void Slam::initialize()
{
    auto anchor_image = m_video_loader.get_next_frame();
    if (anchor_image.empty()) {
        std::cout << "Initialization failed\n";
        return;
    }
    auto anchor_features = m_feature_extractor->extract_features(anchor_image, m_static_mask);
    size_t anchor_index = m_frame_index++;

    std::vector<std::vector<cv::Point2f>> chains(anchor_features.keypoints.size());
    std::vector<cv::Point2f> current;
    std::vector<int> origin;
    current.reserve(anchor_features.keypoints.size());
    origin.reserve(anchor_features.keypoints.size());
    for (size_t i = 0; i < anchor_features.keypoints.size(); i++) {
        current.push_back(anchor_features.keypoints[i].pt);
        origin.push_back(i);
        chains[i].push_back(anchor_features.keypoints[i].pt);
    }

    cv::Mat previous_gray = to_gray(anchor_image);
    std::shared_ptr<Frame> ref_frame;
    std::shared_ptr<Frame> query_frame;
    std::vector<FeatureMatch> accepted_matches;

    while (true) {
        auto image = m_video_loader.get_next_frame();
        if (image.empty()) {
            std::cout << "Initialization failed\n";
            return;
        }
        size_t frame_index = m_frame_index++;
        cv::Mat gray = to_gray(image);

        std::vector<cv::Point2f> next;
        std::vector<cv::Point2f> back;
        std::vector<uchar> ok_forward;
        std::vector<uchar> ok_backward;
        auto window = cv::Size(KLT_WINDOW, KLT_WINDOW);
        cv::calcOpticalFlowPyrLK(
            previous_gray, gray, current, next, ok_forward, cv::noArray(), window, KLT_PYRAMID_LEVELS);
        cv::calcOpticalFlowPyrLK(
            gray, previous_gray, next, back, ok_backward, cv::noArray(), window, KLT_PYRAMID_LEVELS);
        std::vector<cv::Point2f> alive;
        std::vector<int> alive_origin;
        alive.reserve(current.size());
        alive_origin.reserve(origin.size());
        for (size_t i = 0; i < current.size(); i++) {
            if (!ok_forward[i] || !ok_backward[i] || cv::norm(current[i] - back[i]) > KLT_MAX_FORWARD_BACKWARD_ERROR ||
                next[i].x < 0 || next[i].y < 0 || next[i].x >= gray.cols || next[i].y >= gray.rows) {
                continue;
            }
            alive.push_back(next[i]);
            alive_origin.push_back(origin[i]);
            chains[origin[i]].push_back(next[i]);
        }
        current = std::move(alive);
        origin = std::move(alive_origin);
        previous_gray = gray;
        size_t span = frame_index - anchor_index;

        // A starved or overlong epoch re-anchors on the current frame
        if (origin.size() < 200 || span > 30) {
            std::cout << "Init epoch restarted at frame " << frame_index << '\n';
            anchor_features = m_feature_extractor->extract_features(image, m_static_mask);
            anchor_image = image;
            anchor_index = frame_index;
            chains.assign(anchor_features.keypoints.size(), {});
            current.clear();
            origin.clear();
            for (size_t i = 0; i < anchor_features.keypoints.size(); i++) {
                current.push_back(anchor_features.keypoints[i].pt);
                origin.push_back(i);
                chains[i].push_back(anchor_features.keypoints[i].pt);
            }
            continue;
        }

        if (span < 4) {
            continue;
        }
        std::vector<float> displacement;
        displacement.reserve(current.size());
        for (size_t i = 0; i < current.size(); i++) {
            displacement.push_back(cv::norm(current[i] - chains[origin[i]].front()));
        }
        std::nth_element(displacement.begin(), displacement.begin() + displacement.size() / 2, displacement.end());
        if (displacement[displacement.size() / 2] < 20.0F) {
            continue;
        }

        ExtractedFeatures current_features;
        std::vector<FeatureMatch> chain_matches;
        current_features.keypoints.reserve(current.size());
        chain_matches.reserve(current.size());
        for (size_t i = 0; i < current.size(); i++) {
            auto keypoint = anchor_features.keypoints[origin[i]];
            keypoint.pt = current[i];
            chain_matches.emplace_back(origin[i], static_cast<int>(current_features.keypoints.size()));
            current_features.keypoints.push_back(keypoint);
            current_features.descriptors.push_back(anchor_features.descriptors.row(origin[i]));
        }
        auto pose_estimate = pose::estimate_pose(anchor_features, current_features, chain_matches, m_camera);
        std::vector<Eigen::Vector2f> anchor_points;
        std::vector<Eigen::Vector2f> current_points;
        anchor_points.reserve(pose_estimate.inlier_matches.size());
        current_points.reserve(pose_estimate.inlier_matches.size());
        for (const auto& match : pose_estimate.inlier_matches) {
            auto a = anchor_features.keypoints[match.train_index].pt;
            auto c = current_features.keypoints[match.query_index].pt;
            anchor_points.emplace_back(a.x, a.y);
            current_points.emplace_back(c.x, c.y);
        }
        auto candidates = triangulation::triangulate_points(
            anchor_points, current_points, Eigen::Matrix4f::Identity(), pose_estimate.pose, m_camera);
        if (candidates.size() < 100) {
            continue;
        }

        // Third view check: PnP of the chains' middle observations against the candidate map
        size_t middle = span / 2;
        std::vector<cv::Point3f> object_points;
        std::vector<cv::Point2f> image_points;
        object_points.reserve(candidates.size());
        image_points.reserve(candidates.size());
        for (const auto& point : candidates) {
            const auto& match = pose_estimate.inlier_matches[point.match_index];
            const auto& chain = chains[match.train_index];
            if (chain.size() <= middle) {
                continue;
            }
            object_points.emplace_back(point.position.x(), point.position.y(), point.position.z());
            image_points.push_back(chain[middle]);
        }
        if (object_points.size() < 100) {
            continue;
        }
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
                                         2.0F,
                                         0.99,
                                         inliers,
                                         cv::SOLVEPNP_EPNP);
        float support = solved ? static_cast<float>(inliers.rows) / static_cast<float>(object_points.size()) : 0.0F;
        std::cout << "Init candidate " << anchor_index << "->" << frame_index << ": " << candidates.size()
                  << " points, third view support " << support << '\n';
        if (!solved || inliers.rows < 80 || support < 0.6F) {
            continue;
        }

        ref_frame = std::make_shared<Frame>(
            static_cast<int>(anchor_index), anchor_image, anchor_features);
        query_frame =
            std::make_shared<Frame>(static_cast<int>(frame_index), image, current_features);
        query_frame->set_pose(pose_estimate.pose);
        accepted_matches = pose_estimate.inlier_matches;
        break;
    }

    std::cout << "Initializing frames: " << ref_frame->index() << " and " << query_frame->index() << '\n';
    auto points = triangulation::triangulate_points(*ref_frame, *query_frame, accepted_matches, m_camera);

    // Add points to map
    for (size_t i = 0; i < points.size(); i++) {
        auto match = accepted_matches[points[i].match_index];
        m_map.create_point(points[i].position, *ref_frame, *query_frame, match);
    }
    std::cout << "Number of triangulated points: " << points.size() << '\n';

    // Bundle adjustment
    {
        auto config = optimization::OptimizationConfig{
            .optimize_points = true,
            .frames = {{false, ref_frame.get()}, {true, query_frame.get()}},
        };
        optimization::optimize(config, m_camera, m_map);

        float target = m_config.metric_steps.empty() ? 1.0F : metric_distance(ref_frame->index(), query_frame->index());
        float scale =
            target / (query_frame->pose().block<3, 1>(0, 3) - ref_frame->pose().block<3, 1>(0, 3)).stableNorm();
        std::cout << "Scale: " << scale << '\n';

        auto scaled_pose = query_frame->pose();
        scaled_pose.block<3, 1>(0, 3) = query_frame->pose().block<3, 1>(0, 3) * scale;
        query_frame->set_pose(scaled_pose);
        for (auto& point : m_map) {
            point.set_position(point.position() * scale);
        }
    }

    // Add frames to key frames
    m_key_frames.push_back(ref_frame);
    m_key_frames.push_back(query_frame);
    m_last_frame = query_frame;

    record_pose(*ref_frame);
    record_pose(*query_frame);
}

} // namespace slam
