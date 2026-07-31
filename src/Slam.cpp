#include "Slam.h"

#include <unordered_set>
#include <vector>

#include "Frame.h"
#include "Helpers.h"
#include "Optimization.h"
#include "PoseEstimation.h"
#include "Triangulation.h"
#include "features/FeatureExtractor.h"

namespace slam {

// Key frames whose poses local bundle adjustment may move
static const size_t BA_WINDOW = 10;

// Key frame spacing; no minimum, visual change decides
static const size_t MAX_KEY_FRAME_GAP = 20;
static const size_t MIN_TRACKED_POINTS = 50;

// Lucas-Kanade tracking; reach is about window/2 * 2^levels, and a point survives only if
// tracking it backward lands near its origin
static const int KLT_WINDOW = 21;
static const int KLT_PYRAMID_LEVELS = 4;
static const float KLT_MAX_FORWARD_BACKWARD_ERROR = 1.0f;
static const int KLT_REPLENISH_RADIUS = 5; // Matches the detector's minimum feature distance
static const size_t MAX_TRACKED_FEATURES = 2000;

// Tracks triangulate once their rays subtend ~2 degrees; failing tracks stay live
static const float TRACK_MIN_PARALLAX_COSINE = 0.999848f;
static const float TRACK_MAX_REPROJECTION_ERROR = 4.0f;

// Correspondences needed before a RANSAC pose is trusted over the essential matrix
static const size_t MIN_PNP_POINTS = 15;
static const float PNP_REPROJECTION_ERROR = 4.0f;

// Reviving a dropped track by appearance; strict, since a wrong revival teleports a track
static const float REVIVE_MAX_DISTANCE = 50.0f;   // Hamming, against ORB's 256 bits
static const float REVIVE_RATIO = 0.75f;          // Best must beat second best by this
static const float REVIVE_SEARCH_RADIUS = 120.0f; // px from the median-flow prediction

Slam::Slam(const VideoLoader& video_loader,
           const Camera& camera,
           const cv::Mat& image_mask,
           std::unique_ptr<features::BaseFeatureExtractor> feature_extractor,
           const SlamConfig& config)
    : m_video_loader(video_loader), m_camera(camera), m_static_mask(image_mask),
      m_feature_extractor(std::move(feature_extractor)), m_config(config)
{
}

static cv::Mat to_gray(const cv::Mat& image)
{
    if (image.channels() == 1) {
        return image;
    }
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

std::pair<ExtractedFeatures, std::vector<FeatureMatch>> Slam::track_features(const cv::Mat& image)
{
    cv::Mat prev_gray = to_gray(m_last_frame->image());
    cv::Mat next_gray = to_gray(image);

    const auto& prev_features = m_last_frame->features();
    std::vector<cv::Point2f> prev_points;
    for (const auto& keypoint : prev_features.keypoints) {
        prev_points.push_back(keypoint.pt);
    }

    std::vector<cv::Point2f> next_points, back_points;
    std::vector<uchar> forward_ok, backward_ok;
    auto window = cv::Size(KLT_WINDOW, KLT_WINDOW);
    cv::calcOpticalFlowPyrLK(prev_gray, next_gray, prev_points, next_points, forward_ok,
                             cv::noArray(), window, KLT_PYRAMID_LEVELS);
    cv::calcOpticalFlowPyrLK(next_gray, prev_gray, next_points, back_points, backward_ok,
                             cv::noArray(), window, KLT_PYRAMID_LEVELS);

    ExtractedFeatures features;
    std::vector<FeatureMatch> matches;
    std::vector<size_t> lost;
    std::vector<cv::Point2f> flow;
    cv::Mat replenish_mask = m_static_mask.clone();
    for (size_t i = 0; i < prev_points.size(); i++) {
        if (!forward_ok[i] || !backward_ok[i] ||
            cv::norm(prev_points[i] - back_points[i]) > KLT_MAX_FORWARD_BACKWARD_ERROR) {
            lost.push_back(i);
            continue;
        }
        auto point = cv::Point(cvRound(next_points[i].x), cvRound(next_points[i].y));
        if (point.x < 0 || point.y < 0 || point.x >= next_gray.cols ||
            point.y >= next_gray.rows || m_static_mask.at<uchar>(point) == 0) {
            continue;
        }
        flow.push_back(next_points[i] - prev_points[i]);

        auto keypoint = prev_features.keypoints[i];
        keypoint.pt = next_points[i];
        matches.push_back(FeatureMatch(i, features.keypoints.size()));
        features.keypoints.push_back(keypoint);
        features.descriptors.push_back(prev_features.descriptors.row(i));
        cv::circle(replenish_mask, point, KLT_REPLENISH_RADIUS, 0, -1);
    }

    // Refill where nothing is tracked so coverage does not decay as points leave the image;
    // the detector returns corners strongest first, so capping the total keeps the best
    auto new_features = m_feature_extractor->extract_features(image, replenish_mask);

    // Revive tracks flow dropped, by appearance: flow models a patch as pure translation,
    // which fails in a tight turn. A revived point keeps its identity, so the track and
    // everything anchored to it survive
    std::vector<bool> claimed(new_features.keypoints.size(), false);
    size_t revived = 0;
    if (!lost.empty() && !new_features.keypoints.empty() && flow.size() >= 8) {
        // Image motion is rotation-dominated, so the median surviving flow predicts where a
        // lost point should have landed
        std::vector<float> dx, dy;
        for (const auto& f : flow) {
            dx.push_back(f.x);
            dy.push_back(f.y);
        }
        std::nth_element(dx.begin(), dx.begin() + dx.size() / 2, dx.end());
        std::nth_element(dy.begin(), dy.begin() + dy.size() / 2, dy.end());
        cv::Point2f median(dx[dx.size() / 2], dy[dy.size() / 2]);

        cv::Mat lost_descriptors;
        for (auto i : lost) {
            lost_descriptors.push_back(prev_features.descriptors.row(i));
        }
        std::vector<std::vector<cv::DMatch>> knn;
        cv::BFMatcher(cv::NORM_HAMMING).knnMatch(lost_descriptors, new_features.descriptors,
                                                 knn, 2);
        for (size_t k = 0; k < knn.size(); k++) {
            if (knn[k].size() < 2 || knn[k][0].distance > REVIVE_MAX_DISTANCE ||
                knn[k][0].distance > REVIVE_RATIO * knn[k][1].distance) {
                continue;
            }
            int candidate = knn[k][0].trainIdx;
            if (claimed[candidate]) {
                continue;
            }
            auto predicted = prev_points[lost[k]] + median;
            if (cv::norm(new_features.keypoints[candidate].pt - predicted) >
                REVIVE_SEARCH_RADIUS) {
                continue;
            }
            claimed[candidate] = true;
            matches.push_back(FeatureMatch(lost[k], features.keypoints.size()));
            features.keypoints.push_back(new_features.keypoints[candidate]);
            features.descriptors.push_back(new_features.descriptors.row(candidate));
            revived++;
        }
    }

    size_t budget = features.keypoints.size() < MAX_TRACKED_FEATURES
                        ? MAX_TRACKED_FEATURES - features.keypoints.size()
                        : 0;
    for (size_t i = 0; i < new_features.keypoints.size() && budget > 0; i++) {
        if (claimed[i]) {
            continue;
        }
        features.keypoints.push_back(new_features.keypoints[i]);
        features.descriptors.push_back(new_features.descriptors.row(i));
        budget--;
    }

    // Re-describe against this frame so descriptors follow the viewpoint instead of staying
    // frozen at first detection
    features.descriptors = m_feature_extractor->refresh_descriptors(image, features);

    std::cout << "Tracked features: " << matches.size() << " of " << prev_points.size()
              << ", revived " << revived << " of " << lost.size() << ", replenished "
              << new_features.keypoints.size() << std::endl;
    return {features, matches};
}

void Slam::initialize()
{
    // Quality gated initialization: the map is founded only on a frame pair whose
    // triangulation survives a third view. KLT chains from an anchor frame, and a candidate
    // pair is accepted when PnP of the chain's middle observations against the candidate map
    // reaches consensus - sliding tracks fit two views but cannot fake three
    auto anchor_image = m_video_loader.get_next_frame();
    if (anchor_image.empty()) {
        std::cout << "Initialization failed" << std::endl;
        return;
    }
    auto anchor_features = m_feature_extractor->extract_features(anchor_image, m_static_mask);
    size_t anchor_index = m_frame_index++;

    std::vector<std::vector<cv::Point2f>> chains(anchor_features.keypoints.size());
    std::vector<cv::Point2f> current;
    std::vector<int> origin;
    for (size_t i = 0; i < anchor_features.keypoints.size(); i++) {
        current.push_back(anchor_features.keypoints[i].pt);
        origin.push_back(i);
        chains[i].push_back(anchor_features.keypoints[i].pt);
    }

    cv::Mat previous_gray = to_gray(anchor_image);
    std::shared_ptr<Frame> ref_frame, query_frame;
    std::vector<FeatureMatch> accepted_matches;

    while (true) {
        auto image = m_video_loader.get_next_frame();
        if (image.empty()) {
            std::cout << "Initialization failed" << std::endl;
            return;
        }
        size_t frame_index = m_frame_index++;
        cv::Mat gray = to_gray(image);

        std::vector<cv::Point2f> next, back;
        std::vector<uchar> ok_forward, ok_backward;
        auto window = cv::Size(KLT_WINDOW, KLT_WINDOW);
        cv::calcOpticalFlowPyrLK(previous_gray, gray, current, next, ok_forward, cv::noArray(),
                                 window, KLT_PYRAMID_LEVELS);
        cv::calcOpticalFlowPyrLK(gray, previous_gray, next, back, ok_backward, cv::noArray(),
                                 window, KLT_PYRAMID_LEVELS);
        std::vector<cv::Point2f> alive;
        std::vector<int> alive_origin;
        for (size_t i = 0; i < current.size(); i++) {
            if (!ok_forward[i] || !ok_backward[i] ||
                cv::norm(current[i] - back[i]) > KLT_MAX_FORWARD_BACKWARD_ERROR ||
                next[i].x < 0 || next[i].y < 0 || next[i].x >= gray.cols ||
                next[i].y >= gray.rows) {
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
            std::cout << "Init epoch restarted at frame " << frame_index << std::endl;
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
        for (size_t i = 0; i < current.size(); i++) {
            displacement.push_back(cv::norm(current[i] - chains[origin[i]].front()));
        }
        std::nth_element(displacement.begin(), displacement.begin() + displacement.size() / 2,
                         displacement.end());
        if (displacement[displacement.size() / 2] < 20.0f) {
            continue;
        }

        ExtractedFeatures current_features;
        std::vector<FeatureMatch> chain_matches;
        for (size_t i = 0; i < current.size(); i++) {
            auto keypoint = anchor_features.keypoints[origin[i]];
            keypoint.pt = current[i];
            chain_matches.push_back(FeatureMatch(origin[i], current_features.keypoints.size()));
            current_features.keypoints.push_back(keypoint);
            current_features.descriptors.push_back(anchor_features.descriptors.row(origin[i]));
        }
        auto pose_estimate =
            pose::estimate_pose(anchor_features, current_features, chain_matches, m_camera);
        std::vector<Eigen::Vector2f> anchor_points, current_points;
        for (const auto& match : pose_estimate.inlier_matches) {
            auto a = anchor_features.keypoints[match.train_index].pt;
            auto c = current_features.keypoints[match.query_index].pt;
            anchor_points.push_back(Eigen::Vector2f(a.x, a.y));
            current_points.push_back(Eigen::Vector2f(c.x, c.y));
        }
        auto candidates = triangulation::triangulate_points(anchor_points,
                                                            current_points,
                                                            Eigen::Matrix4f::Identity(),
                                                            pose_estimate.pose,
                                                            m_camera);
        if (candidates.size() < 100) {
            continue;
        }

        // Third view check: PnP of the chains' middle observations against the candidate map
        size_t middle = span / 2;
        std::vector<cv::Point3f> object_points;
        std::vector<cv::Point2f> image_points;
        for (const auto& point : candidates) {
            const auto& match = pose_estimate.inlier_matches[point.match_index];
            const auto& chain = chains[match.train_index];
            if (chain.size() <= middle) {
                continue;
            }
            object_points.push_back(
                cv::Point3f(point.position.x(), point.position.y(), point.position.z()));
            image_points.push_back(chain[middle]);
        }
        if (object_points.size() < 100) {
            continue;
        }
        cv::Mat intrinsics;
        cv_utils::intrinsic_mat_cv(m_camera).convertTo(intrinsics, CV_64F);
        cv::Mat rvec, tvec, inliers;
        bool solved = cv::solvePnPRansac(object_points, image_points, intrinsics, cv::Mat(),
                                         rvec, tvec, false, 200, 2.0f, 0.99, inliers,
                                         cv::SOLVEPNP_EPNP);
        float support = solved ? float(inliers.rows) / object_points.size() : 0.0f;
        std::cout << "Init candidate " << anchor_index << "->" << frame_index << ": "
                  << candidates.size() << " points, third view support " << support << std::endl;
        if (!solved || inliers.rows < 80 || support < 0.6f) {
            continue;
        }

        ref_frame = std::make_shared<Frame>(anchor_index, anchor_image, anchor_features);
        query_frame = std::make_shared<Frame>(frame_index, image, current_features);
        query_frame->set_pose(pose_estimate.pose);
        accepted_matches = pose_estimate.inlier_matches;
        break;
    }

    std::cout << "Initializing frames: " << ref_frame->index() << " and " << query_frame->index()
              << std::endl;
    auto points =
        triangulation::triangulate_points(*ref_frame, *query_frame, accepted_matches, m_camera);

    // Add points to map
    for (int i = 0; i < points.size(); i++) {
        auto match = accepted_matches[points[i].match_index];
        m_map.create_point(points[i].position, *ref_frame, *query_frame, match);
    }
    std::cout << "Number of triangulated points: " << points.size() << std::endl;

    // Bundle adjustment
    {
        auto config = optimization::OptimizationConfig{
            .optimize_points = true,
            .frames = {{false, ref_frame.get()}, {true, query_frame.get()}},
        };
        optimization::optimize(config, m_camera, m_map);

        float scale =
            1.0f / (query_frame->pose().block<3, 1>(0, 3) - ref_frame->pose().block<3, 1>(0, 3))
                       .stableNorm();
        std::cout << "Scale: " << scale << std::endl;

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

bool Slam::step()
{
    std::cout << "----------------------------------------" << std::endl;
    auto image = m_video_loader.get_next_frame();
    if (image.empty()) {
        std::cout << "No frame to process" << std::endl;
        return false;
    }

    ExtractedFeatures features;
    std::vector<FeatureMatch> tracked;
    time_it("Track features", [&]() { std::tie(features, tracked) = track_features(image); });
    auto frame = std::make_shared<Frame>(m_frame_index++, image, features);
    auto last_key_frame = m_key_frames.back();

    // Initial pose estimation
    std::vector<FeatureMatch> inlier_matches;
    time_it("Initial pose estimation",
            [&]() { inlier_matches = initial_pose_estimate(*frame, tracked); });
    time_it("Update tracks", [&]() { update_tracks(inlier_matches); });
    time_it("Track from last frame", [&]() { track_from_last_frame(*frame, inlier_matches); });
    time_it("Optimize pose", [&]() { optimize_pose(*frame); });

    // Match with map from last key frame
    time_it("Match with last key frame", [&]() { match_with_last_key_frame(*frame); });
    time_it("Optimize pose", [&]() { optimize_pose(*frame); });

    // Match with all map points
    time_it("Match with map", [&]() { match_with_map(*frame); });
    time_it("Optimize pose", [&]() { optimize_pose(*frame); });

    // Create key frame if needed
    time_it("Create key frame", [&]() {
        if (needs_key_frame(*frame, *last_key_frame)) {
            std::cout << "Adding key frame after " << frame->index() - last_key_frame->index()
                      << " frames" << std::endl;
            init_key_frame(*frame);
            m_key_frames.push_back(frame);
        }
    });

    m_last_frame = frame;
    record_pose(*frame);
    return true;
}

bool Slam::needs_key_frame(const Frame& frame, const Frame& last_key_frame) const
{
    if (frame.num_map_matches() < MIN_TRACKED_POINTS) {
        return true;
    }

    size_t gap = frame.index() - last_key_frame.index();
    return gap >= MAX_KEY_FRAME_GAP ||
           frame.num_map_matches() < 0.9 * last_key_frame.num_map_matches();
}

void Slam::record_pose(const Frame& frame)
{
    // Frames dropped during initialization keep the previous pose so that the trajectory stays
    // indexed by frame index
    auto fill = m_trajectory.empty() ? Eigen::Matrix4f::Identity() : m_trajectory.back();
    m_trajectory.resize(frame.index() + 1, fill);
    m_trajectory[frame.index()] = frame.pose();
}

std::vector<FeatureMatch> Slam::initial_pose_estimate(Frame& frame,
                                                      const std::vector<FeatureMatch>& matches)
{
    if (m_config.essential_matrix_estimation || m_key_frames.size() < 2) {
        auto pose_estimate =
            pose::estimate_pose(m_last_frame->features(), frame.features(), matches, m_camera);
        // The essential matrix measures rotation and direction but not step length, so the
        // previous frame's step stands in for its unit magnitude
        auto index = m_last_frame->index();
        if (index >= 1 && index < m_trajectory.size()) {
            auto center = [](const Eigen::Matrix4f& pose) -> Eigen::Vector3f {
                return -pose.block<3, 3>(0, 0).transpose() * pose.block<3, 1>(0, 3);
            };
            float last_step =
                (center(m_trajectory[index]) - center(m_trajectory[index - 1])).norm();
            auto scaled = pose_estimate.pose;
            if (scaled.block<3, 1>(0, 3).norm() > 1e-6f && last_step > 1e-6f) {
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
    for (const auto& match : matches) {
        if (!m_last_frame->is_matched(match.train_index)) {
            continue;
        }
        const auto& point = m_last_frame->map_match(match.train_index);
        // Mint early, trust late: a single-key-frame point exists so coverage never starves,
        // but earns a pose vote only once a second observation confirms it
        if (point.observations().size() < 2) {
            continue;
        }
        object_points.push_back(
            cv::Point3f(point.position().x(), point.position().y(), point.position().z()));
        image_points.push_back(frame.keypoint(match.query_index).pt);
        points.push_back(&point);
        keypoint_indices.push_back(match.query_index);
    }

    if (object_points.size() < MIN_PNP_POINTS) {
        std::cout << "Too few correspondences to track from last frame" << std::endl;
        return;
    }

    // RANSAC rather than least squares: a robust kernel only down-weights wrong matches where
    // consensus excludes them outright
    cv::Mat intrinsics;
    cv_utils::intrinsic_mat_cv(m_camera).convertTo(intrinsics, CV_64F);
    cv::Mat rvec, tvec, inliers;
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
    if (!solved || inliers.rows < (int)MIN_PNP_POINTS) {
        std::cout << "RANSAC pose rejected, keeping essential matrix estimate" << std::endl;
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

    frame.set_pose(pose);

    // Only the consensus set is kept, so bundle adjustment refines clean data
    for (int i = 0; i < inliers.rows; i++) {
        int index = inliers.at<int>(i, 0);
        if (frame.is_matched(keypoint_indices[index]) || frame.is_matched(*points[index])) {
            continue;
        }
        frame.add_map_match(MapPointMatch{*points[index], keypoint_indices[index]});
    }
    std::cout << "Tracked from last frame: " << inliers.rows << " / " << object_points.size()
              << std::endl;
}

void Slam::update_tracks(const std::vector<FeatureMatch>& matches)
{
    // train_index indexes the previous frame, query_index the current one
    std::unordered_map<size_t, FeatureTrack> carried;
    for (const auto& match : matches) {
        auto existing = m_tracks.find(match.train_index);
        carried[match.query_index] =
            existing != m_tracks.end() ? existing->second : FeatureTrack{};
    }
    m_tracks = std::move(carried);
}

void Slam::triangulate_tracks(Frame& frame)
{
    // Group by anchor key frame so every group shares one live pose pair; tracks without an
    // anchor yet have no independent view and wait
    std::unordered_map<size_t, std::vector<size_t>> by_anchor;
    for (const auto& [keypoint_index, track] : m_tracks) {
        if (frame.is_matched(keypoint_index) || track.anchor_key_frame == nullptr) {
            continue;
        }
        by_anchor[track.anchor_key_frame->index()].push_back(keypoint_index);
    }

    size_t created = 0;
    std::vector<size_t> poisoned;
    for (const auto& [anchor_index, keypoint_indices] : by_anchor) {
        const auto& track = m_tracks.at(keypoint_indices.front());
        std::vector<Eigen::Vector2f> anchor_points;
        std::vector<Eigen::Vector2f> current_points;
        for (auto index : keypoint_indices) {
            anchor_points.push_back(m_tracks.at(index).anchor_pixel);
            auto pixel = frame.keypoint(index).pt;
            current_points.push_back(Eigen::Vector2f(pixel.x, pixel.y));
        }

        auto points = triangulation::triangulate_points(anchor_points,
                                                        current_points,
                                                        track.anchor_key_frame->pose(),
                                                        frame.pose(),
                                                        m_camera,
                                                        TRACK_MIN_PARALLAX_COSINE,
                                                        TRACK_MAX_REPROJECTION_ERROR);
        for (const auto& point : points) {
            auto keypoint_index = keypoint_indices[point.match_index];
            // Two views cannot expose a track that slid off its physical point but stayed
            // self-consistent; the observation at the key frame in between can
            const auto& candidate = m_tracks.at(keypoint_index);
            if (candidate.mid_key_frame != candidate.anchor_key_frame) {
                auto mid_error =
                    (m_camera.project(candidate.mid_key_frame->pose(), point.position) -
                     candidate.mid_pixel)
                        .norm();
                if (mid_error > 2) {
                    poisoned.push_back(keypoint_index);
                    continue;
                }
            }
            m_map.create_point(point.position, frame, keypoint_index);
            created++;
        }
    }

    for (auto keypoint_index : poisoned) {
        m_tracks.erase(keypoint_index);
    }

    // Anchor new tracks to this key frame; it becomes the latest observation of the rest
    for (auto& [keypoint_index, track] : m_tracks) {
        if (frame.is_matched(keypoint_index)) {
            continue;
        }
        auto pixel = frame.keypoint(keypoint_index).pt;
        if (track.anchor_key_frame == nullptr) {
            track.anchor_key_frame = &frame;
            track.anchor_pixel = Eigen::Vector2f(pixel.x, pixel.y);
        }
        track.mid_key_frame = &frame;
        track.mid_pixel = Eigen::Vector2f(pixel.x, pixel.y);
    }
    std::cout << "Triangulated from tracks: " << created << " of " << m_tracks.size()
              << " tracks, poisoned " << poisoned.size() << std::endl;
}

void Slam::match_with_last_key_frame(Frame& frame)
{
    auto last_frame = m_key_frames.back();
    auto map_matches =
        m_feature_extractor->match_features(frame, m_camera, m_map, [&](const MapPoint& point) {
            return point.is_observed_by(last_frame.get());
        });
    for (const auto& match : map_matches) {
        frame.add_map_match(match);
    }
    std::cout << "Map matches with last frame: " << map_matches.size() << std::endl;
}

void Slam::match_with_map(Frame& frame)
{
    auto map_matches = m_feature_extractor->match_features(
        frame, m_camera, m_map, [&](const MapPoint& point) { return true; });
    for (const auto& match : map_matches) {
        frame.add_map_match(match);
    }
    std::cout << "Number of map matches: " << map_matches.size() << std::endl;
}

void Slam::optimize_pose(Frame& frame)
{
    if (!m_config.optimize_pose) {
        return;
    }
    // A pose solve against a handful of matches can move the pose anywhere that fits them;
    // below the floor the current estimate stands
    if (frame.num_map_matches() < MIN_PNP_POINTS) {
        return;
    }
    // Motion-only BA
    auto config = optimization::OptimizationConfig{
        .optimize_points = false,
        .frames = {{true, &frame}},
    };
    optimization::optimize(config, m_camera, m_map);
}

void Slam::init_key_frame(Frame& frame)
{
    // Add map associations
    for (const auto& match : frame.map_matches()) {
        m_map.add_association(frame, match);
    }

    // Triangulate from feature tracks rather than key frame to key frame matches
    if (m_config.triangulate_points) {
        triangulate_tracks(frame);
    }

    // Local bundle adjustment (covisbility graph)
    if (m_config.bundle_adjust) {
        size_t first_optimized =
            m_key_frames.size() > BA_WINDOW ? m_key_frames.size() - BA_WINDOW : 2;
        std::unordered_set<const Frame*> window{&frame};
        for (size_t i = first_optimized; i < m_key_frames.size(); i++) {
            window.insert(m_key_frames[i].get());
        }

        // Add any frames that share map point observations with the window
        std::unordered_set<const Frame*> anchors;
        for (const Frame* window_frame : window) {
            for (const auto& match : window_frame->map_matches()) {
                for (const auto& [observer, _] : match.point.observations()) {
                    if (window.find(observer) == window.end()) {
                        anchors.insert(observer);
                    }
                }
            }
        }

        // Build optimization config
        std::vector<optimization::FrameConfig> frame_configs;
        for (const auto& key_frame : m_key_frames) {
            if (window.find(key_frame.get()) != window.end()) {
                frame_configs.push_back({true, key_frame.get()});
            } else if (anchors.find(key_frame.get()) != anchors.end()) {
                frame_configs.push_back({false, key_frame.get()});
            }
        }
        frame_configs.push_back({true, &frame});
        auto config = optimization::OptimizationConfig{
            .optimize_points = true,
            .frames = frame_configs,
        };
        optimization::optimize(config, m_camera, m_map);
    }

    // Cull points
    if (m_config.cull_points) {
        cull_points();
    }
}

void Slam::cull_points()
{
    std::vector<MapPoint*> points_to_remove;
    for (auto& point : m_map) {
        float error = 0.0;
        int num_projected = 0;
        for (const auto& [frame, index] : point.observations()) {
            auto projected = m_camera.project(frame->pose(), point.position());
            auto image_point =
                Eigen::Vector2f(frame->keypoint(index).pt.x, frame->keypoint(index).pt.y);
            error += (projected - image_point).norm();
            num_projected++;
        }
        if (error / num_projected > 3.0) {
            points_to_remove.push_back(&point);
        }
    }

    std::cout << "Number of points to remove: " << points_to_remove.size() << std::endl;
    for (const auto& point : points_to_remove) {
        m_map.remove_point(point);
    }
}

float Slam::reprojection_error() const
{
    float error = 0.0;
    int num_projected = 0;
    for (const auto& frame : m_key_frames) {
        for (const auto& match : frame->map_matches()) {
            auto point = match.point;
            auto projected = m_camera.project(frame->pose(), point.position());
            auto image_point = Eigen::Vector2f(frame->keypoint(match.keypoint_index).pt.x,
                                               frame->keypoint(match.keypoint_index).pt.y);
            error += (projected - image_point).stableNorm();
            num_projected++;
        }
    }
    return error / num_projected;
}

const Map& Slam::map() const
{
    return m_map;
}

const Frame& Slam::frame() const
{
    return *m_last_frame;
}

std::vector<Eigen::Matrix4f> Slam::poses() const
{
    std::vector<Eigen::Matrix4f> poses;
    for (const auto& frame : m_key_frames) {
        poses.push_back(frame->pose());
    }
    return poses;
}

std::vector<Eigen::Matrix4f> Slam::trajectory() const
{
    // Key frames are refined by bundle adjustment after they were recorded
    auto trajectory = m_trajectory;
    for (const auto& frame : m_key_frames) {
        if (frame->index() < trajectory.size()) {
            trajectory[frame->index()] = frame->pose();
        }
    }
    return trajectory;
}

} // namespace slam
