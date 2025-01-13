#include "Slam.h"

#include "Features.h"
#include "Init.h"
#include "Optimization.h"
#include "PoseEstimation.h"
#include "Triangulation.h"

namespace slam {

Slam::Slam(const VideoLoader& video_loader,
           const Camera& camera,
           const cv::Mat& image_mask,
           const SlamConfig& config)
    : m_video_loader(video_loader), m_camera(camera), m_static_mask(image_mask), m_config(config)
{
}

std::optional<Frame> Slam::process_next_frame()
{
    auto image = m_video_loader.get_next_frame();
    if (image.empty()) {
        return std::nullopt;
    }
    auto features = features::extract_features(image, m_static_mask);
    return std::make_optional<Frame>(m_frame_index++, image, features);
}

void Slam::initialize()
{
    // First first frames
    auto maybe_result = init::find_initializing_frames([this]() { return process_next_frame(); },
                                                       m_camera);
    if (!maybe_result) {
        std::cout << "Initialization failed" << std::endl;
        return;
    }

    auto result = std::move(*maybe_result);
    auto ref_frame = std::make_shared<Frame>(std::move(result.ref_frame));
    auto query_frame = std::make_shared<Frame>(std::move(result.query_frame));
    std::cout << "Initializing frames: " << ref_frame->index() << " and " << query_frame->index()
              << std::endl;

    auto matches = features::match_features(ref_frame->features(), query_frame->features());

    // Triangulate points
    auto points = triangulation::triangulate_points(*ref_frame, *query_frame, matches, m_camera);

    // Add points to map
    for (int i = 0; i < points.size(); i++) {
        auto train_index = result.matches[points[i].match_index].train_index;
        auto query_index = result.matches[points[i].match_index].query_index;

        auto point = std::make_unique<MapPoint>(points[i].position);
        point->add_observation(ref_frame.get(), train_index);
        point->add_observation(query_frame.get(), query_index);
        ref_frame->add_map_match(MapPointMatch{*point, train_index});
        query_frame->add_map_match(MapPointMatch{*point, query_index});
        m_map.add_point(std::move(point));
    }
    std::cout << "Number of triangulated points: " << points.size() << std::endl;

    // Bundle adjustment
    {
        auto config = optimization::OptimizationConfig{
            .optimize_points = true,
            .frames = {{false, ref_frame.get()}, {true, query_frame.get()}},
        };
        optimization::optimize(config, m_camera, m_map);

        float scale = 1.0f /
                      (query_frame->pose().block<3, 1>(0, 3) - ref_frame->pose().block<3, 1>(0, 3))
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
    m_frame = query_frame;
}

void Slam::step()
{
    auto maybe_frame = process_next_frame();
    if (!maybe_frame) {
        std::cout << "No frame to process" << std::endl;
        return;
    }

    const auto& last_frame = m_key_frames.back();
    auto frame = std::make_shared<Frame>(std::move(*maybe_frame));

    // Initial pose estimation
    if (m_config.essential_matrix_estimation) {
        auto pose_estimate = pose::estimate_pose(
            last_frame->features(),
            frame->features(),
            features::match_features(last_frame->features(), frame->features()),
            m_camera);
        frame->set_pose(pose_estimate.pose * last_frame->pose());
    } else {
        frame->set_pose(last_frame->pose());
    }

    // Match with map
    auto map_matches = features::match_features(*frame, m_camera, m_map);
    for (const auto& match : map_matches) {
        frame->add_map_match(match);
        const_cast<MapPoint&>(match.point).add_observation(frame.get(), match.keypoint_index);
    }
    std::cout << "Number of map matches: " << map_matches.size() << std::endl;

    // Optimize the pose
    if (m_config.optimize_pose) {
        auto config = optimization::OptimizationConfig{
            .optimize_points = false,
            .frames = {{true, frame.get()}},
        };
        optimization::optimize(config, m_camera, m_map);
    }

    // // Triangulate unmatched points
    if (m_config.triangulate_points) {
        auto feature_matches = features::match_features(last_frame->features(), frame->features());
        auto unmatched = features::unmatched_features(*last_frame, *frame, feature_matches);
        auto points = triangulation::triangulate_points(*last_frame, *frame, unmatched, m_camera);
        for (int i = 0; i < points.size(); i++) {
            auto train_index = unmatched[points[i].match_index].train_index;
            auto query_index = unmatched[points[i].match_index].query_index;
            auto point = std::make_unique<MapPoint>(points[i].position);
            point->add_observation(last_frame.get(), train_index);
            point->add_observation(frame.get(), query_index);
            last_frame->add_map_match(MapPointMatch{*point, train_index});
            frame->add_map_match(MapPointMatch{*point, query_index});
            m_map.add_point(std::move(point));
        }
        std::cout << "Number of triangulated points: " << points.size() << std::endl;
    }

    // Bundle adjustment
    if (m_config.bundle_adjust) {
        std::vector<optimization::FrameConfig> frame_configs;
        for (const auto& frame : m_key_frames) {
            frame_configs.push_back({false, frame.get()});
        }
        frame_configs.push_back({true, frame.get()});
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

    // Add frame to key frames
    m_key_frames.push_back(frame);
    m_frame = frame;
}

void Slam::cull_points()
{
    std::vector<MapPoint*> points_to_remove;
    for (auto& point : m_map) {
        float error = 0.0;
        int num_projected = 0;
        for (const auto& [frame, index] : point.observations()) {
            auto projected = m_camera.project(frame->pose(), point.position());
            auto image_point = Eigen::Vector2f(frame->keypoint(index).pt.x,
                                               frame->keypoint(index).pt.y);
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
    return *m_frame;
}

std::vector<Eigen::Matrix4f> Slam::poses() const
{
    std::vector<Eigen::Matrix4f> poses;
    for (const auto& frame : m_key_frames) {
        poses.push_back(frame->pose());
    }
    return poses;
}

} // namespace slam