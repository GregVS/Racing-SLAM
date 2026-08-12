#include "Tracker.h"

#include "Imu.h"

#include "Frame.h"
#include "Helpers.h"
#include "MotionModel.h"
#include "Optimization.h"
#include "PoseEstimation.h"
#include "Slam.h"
#include "features/FeatureExtractor.h"

namespace slam {

namespace {

constexpr int KLT_WINDOW = 21;
constexpr int KLT_PYRAMID_LEVELS = 4;
constexpr float KLT_MAX_FORWARD_BACKWARD_ERROR = 1.0F;
constexpr int KLT_REPLENISH_RADIUS = 5;
constexpr size_t MAX_TRACKED_FEATURES = 2000;
constexpr size_t MIN_TRACKED_MAP_POINTS = 15;

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

Tracker::Tracker(const Camera& camera,
                 const cv::Mat& static_mask,
                 const features::BaseFeatureExtractor& feature_extractor,
                 const SlamConfig& config,
                 Map& map)
    : m_camera(camera), m_static_mask(static_mask), m_feature_extractor(feature_extractor), m_config(config),
      m_map(map), m_map_matcher(camera, feature_extractor.max_distance(), feature_extractor.norm_type())
{
}

TrackStore& Tracker::tracks()
{
    return m_tracks;
}

Frame& Tracker::last_frame() const
{
    return *m_last_frame;
}

bool Tracker::has_last_frame() const
{
    return m_last_frame != nullptr;
}

void Tracker::set_last_frame(const std::shared_ptr<Frame>& frame)
{
    m_last_frame = frame;
}

std::shared_ptr<Frame> Tracker::track(const cv::Mat& image,
                                      size_t frame_index,
                                      const Trajectory& trajectory,
                                      KeyFrame& last_key_frame,
                                      size_t num_key_frames)
{
    m_last_key_frame = &last_key_frame;

    ExtractedFeatures features;
    std::vector<FeatureMatch> tracked;
    time_it("Track features", [&]() { std::tie(features, tracked) = track_features(image); });
    auto frame = std::make_shared<Frame>(frame_index, image, features);

    std::vector<FeatureMatch> inlier_matches;
    time_it("Initial pose estimation",
            [&]() { inlier_matches = initial_pose_estimate(*frame, tracked, trajectory, num_key_frames); });
    time_it("Update tracks", [&]() { m_tracks.carry_forward(inlier_matches); });
    time_it("Track from last frame", [&]() { track_from_last_frame(*frame, inlier_matches); });
    time_it("Optimize pose", [&]() { optimize_pose(*frame); });
    time_it("Match with last key frame", [&]() { match_with_last_key_frame(*frame, last_key_frame); });
    time_it("Match with map", [&]() { match_with_map(*frame); });
    return frame;
}

std::pair<ExtractedFeatures, std::vector<FeatureMatch>> Tracker::track_features(const cv::Mat& image)
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
    auto new_features = m_feature_extractor.extract_features(image, replenish_mask);

    size_t budget =
        features.keypoints.size() < MAX_TRACKED_FEATURES ? MAX_TRACKED_FEATURES - features.keypoints.size() : 0;
    for (size_t i = 0; i < new_features.keypoints.size() && budget > 0; i++) {
        features.keypoints.push_back(new_features.keypoints[i]);
        features.descriptors.push_back(new_features.descriptors.row(i));
        budget--;
    }

    // Re-describe against this frame so descriptors follow the viewpoint instead of staying
    // frozen at first detection
    features.descriptors = m_feature_extractor.refresh_descriptors(image, features);

    std::cout << "Tracked features: " << matches.size() << " of " << prev_points.size() << ", replenished "
              << new_features.keypoints.size() << '\n';
    return {std::move(features), std::move(matches)};
}

std::vector<FeatureMatch> Tracker::initial_pose_estimate(Frame& frame,
                                                         const std::vector<FeatureMatch>& matches,
                                                         const Trajectory& trajectory,
                                                         size_t num_key_frames)
{
    if (m_config.essential_matrix_estimation || num_key_frames < 2) {
        pose::PoseEstimate pose_estimate =
            pose::estimate_pose(m_last_frame->features(), frame.features(), matches, m_camera);
        auto index = m_last_frame->index();
        if (index >= 1 && index < trajectory.size()) {
            float last_step = (motion::camera_center(trajectory.pose_at(index)) -
                               motion::camera_center(trajectory.pose_at(index - 1)))
                                  .norm();
            Eigen::Matrix4f relative = pose_estimate.pose;
            Eigen::Matrix4f candidate = relative * m_last_frame->pose();
            if (!motion::is_rotation_plausible(m_last_frame->pose(), candidate, m_config.seconds_per_frame)) {
                std::cout << "Essential rotation rejected by temporal motion bound\n";
                relative.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
            }

            auto scaled = relative;
            if (scaled.block<3, 1>(0, 3).norm() > 1e-6F && last_step > 1e-6F) {
                scaled.block<3, 1>(0, 3) *= last_step / scaled.block<3, 1>(0, 3).norm();
                frame.set_pose(scaled * m_last_frame->pose());
                if (m_config.inertial_pose_seed) {
                    seed_pose_from_inertial(frame, last_step);
                }
                return pose_estimate.inlier_matches;
            }
        }
        frame.set_pose(pose_estimate.pose * m_last_frame->pose());
        if (m_config.inertial_pose_seed) {
            seed_pose_from_inertial(frame, 0.0F);
        }
        return pose_estimate.inlier_matches;
    }
    frame.set_pose(m_last_frame->pose());
    return {};
}

void Tracker::track_from_last_frame(Frame& frame, const std::vector<FeatureMatch>& matches)
{
    // Carry over map points from the previous frame that are still visible in the current frame
    std::vector<MapPoint*> points;
    std::vector<size_t> keypoint_indices;
    points.reserve(matches.size());
    keypoint_indices.reserve(matches.size());
    for (const auto& match : matches) {
        if (!m_last_frame->is_matched(match.train_index)) {
            continue;
        }
        auto& point = m_last_frame->map_match(match.train_index);
        if (point.observations().size() < 2 && !point.track_consistent()) {
            continue;
        }
        points.push_back(&point);
        keypoint_indices.push_back(match.query_index);
    }

    if (points.size() < MIN_TRACKED_MAP_POINTS) {
        std::cout << "Too few correspondences to track from last frame\n";
        return;
    }

    size_t accepted = 0;
    for (size_t i = 0; i < points.size(); i++) {
        if (frame.is_matched(keypoint_indices[i]) || frame.is_matched(*points[i])) {
            continue;
        }
        frame.add_map_match(MapPointMatch{*points[i], keypoint_indices[i]});
        accepted++;
    }
    std::cout << "Tracked from last frame: " << accepted << " / " << points.size() << '\n';
}

void Tracker::match_with_last_key_frame(Frame& frame, KeyFrame& last_key_frame)
{
    auto map_matches = m_map_matcher.match_key_frame(frame, m_map, &last_key_frame);
    for (const auto& match : map_matches) {
        frame.add_map_match(match);
    }
    std::cout << "Map matches with last frame: " << map_matches.size() << '\n';
}

void Tracker::match_with_map(Frame& frame)
{
    auto map_matches = m_map_matcher.match_map(frame, m_map);
    for (const auto& match : map_matches) {
        frame.add_map_match(match);
    }
    std::cout << "Number of map matches: " << map_matches.size() << '\n';
}

optimization::RotationPrior Tracker::rotation_prior(const Frame& frame) const
{
    optimization::RotationPrior prior;
    if (m_inertial == nullptr || m_inertial_input == nullptr || m_config.seconds_per_frame <= 0.0F ||
        m_last_frame == nullptr) {
        return prior;
    }
    const double from = static_cast<double>(m_last_frame->index()) * m_config.seconds_per_frame;
    const double to = static_cast<double>(frame.index()) * m_config.seconds_per_frame;
    const std::vector<imu::Sample> samples = m_inertial->between(from, to);
    if (samples.size() < 2 || to <= from) {
        return prior;
    }

    prior.predicted =
        imu::integrate_rotation(samples).transpose() * m_last_frame->pose().block<3, 3>(0, 0).cast<double>();
    prior.sigma_radians = m_inertial_input->attitude_error_density * std::sqrt(to - from);
    return prior;
}

optimization::InertialDelta Tracker::inertial_step(const Frame& frame) const
{
    optimization::InertialDelta step;
    if (m_inertial_input == nullptr || !m_inertial_input->usable() || m_last_frame == nullptr) {
        return step;
    }
    const Frame* anchor = m_last_key_frame != nullptr && m_last_key_frame->index() < frame.index() ? m_last_key_frame
                                                                                                   : m_last_frame.get();
    const double from = m_inertial_input->time_of(anchor->index());
    const double to = m_inertial_input->time_of(frame.index());
    const std::vector<imu::Sample> samples = m_inertial_input->stream->between(from, to);
    if (samples.size() < 2) {
        return step;
    }
    step.previous = anchor;
    step.summary = imu::preintegrate(samples, m_inertial_input->noise, anchor->inertial().bias);
    step.gravity = m_inertial_input->gravity;
    step.noise = m_inertial_input->noise;
    return step;
}

optimization::InertialConstraint Tracker::inertial_constraint(const Frame& frame) const
{
    if (const optimization::InertialDelta step = inertial_step(frame); step.enabled()) {
        return step;
    }
    if (const optimization::RotationPrior prior = rotation_prior(frame); prior.enabled()) {
        return prior;
    }
    return {};
}

void Tracker::optimize_pose(Frame& frame)
{
    if (!m_config.optimize_pose) {
        return;
    }
    if (frame.num_map_matches() < MIN_TRACKED_MAP_POINTS) {
        return;
    }

    Eigen::Matrix4f before = frame.pose();

    bool optimized = optimization::refine_pose(frame, m_camera, inertial_constraint(frame));
    if (!optimized || !motion::is_rotation_plausible(m_last_frame->pose(), frame.pose(), m_config.seconds_per_frame)) {
        frame.set_pose(before);
        if (optimized) {
            std::cout << "Pose optimization rolled back by temporal motion bound\n";
        }
    }
}

} // namespace slam

namespace slam {

void Tracker::set_inertial(const imu::Stream* stream)
{
    m_inertial = stream;
}

void Tracker::set_inertial_input(const optimization::InertialInput* input)
{
    m_inertial_input = input;
}

bool Tracker::seed_pose_from_inertial(Frame& frame, float step_length)
{
    if (m_inertial == nullptr || m_config.seconds_per_frame <= 0.0F || m_last_frame == nullptr) {
        return false;
    }
    const double from = static_cast<double>(m_last_frame->index()) * m_config.seconds_per_frame;
    const double to = static_cast<double>(frame.index()) * m_config.seconds_per_frame;
    const std::vector<imu::Sample> samples = m_inertial->between(from, to);
    if (samples.size() < 2) {
        return false;
    }

    const Frame* anchor = m_last_key_frame != nullptr && m_last_key_frame->index() < frame.index() ? m_last_key_frame
                                                                                                   : m_last_frame.get();
    if (m_inertial_input != nullptr && m_inertial_input->usable()) {
        // IMU is aligned
        const std::vector<imu::Sample> span =
            m_inertial->between(m_inertial_input->time_of(anchor->index()), m_inertial_input->time_of(frame.index()));
        if (span.size() >= 2) {
            const imu::Preintegrated summary =
                imu::preintegrate(span, m_inertial_input->noise, anchor->inertial().bias);
            set_inertial_state(frame, imu::predict(inertial_state(*anchor), summary, m_inertial_input->gravity));
            return true;
        }
    }

    // IMU is not aligned yet, only rotation can be used
    if (step_length <= 0.0F) {
        return false;
    }
    const Eigen::Matrix3f previous_world_to_camera = m_last_frame->pose().block<3, 3>(0, 0);
    const Eigen::Matrix3f increment = imu::integrate_rotation(samples).cast<float>();
    const Eigen::Matrix3f world_to_camera = increment.transpose() * previous_world_to_camera;
    const Eigen::Vector3f forward = previous_world_to_camera.transpose().col(2);
    const Eigen::Vector3f center = m_last_frame->camera_center() + step_length * forward;

    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    pose.block<3, 3>(0, 0) = world_to_camera;
    pose.block<3, 1>(0, 3) = -world_to_camera * center;
    frame.set_pose(pose);
    return true;
}

} // namespace slam
