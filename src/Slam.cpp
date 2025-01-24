#include "Slam.h"

#include "features/FeatureExtractor.h"
#include "Helpers.h"
#include "Init.h"
#include "Optimization.h"
#include "PoseEstimation.h"
#include "Triangulation.h"

namespace slam {

Slam::Slam(const VideoLoader& video_loader,
           const Camera& camera,
           const cv::Mat& image_mask,
           std::unique_ptr<features::BaseFeatureExtractor> feature_extractor,
           const SlamConfig& config)
    : m_video_loader(video_loader), m_camera(camera), m_static_mask(image_mask),
      m_feature_extractor(std::move(feature_extractor)), m_config(config)
{
}

std::optional<Frame> Slam::process_next_frame()
{
    auto image = m_video_loader.get_next_frame();
    if (image.empty()) {
        return std::nullopt;
    }
    auto features = m_feature_extractor->extract_features(image, m_static_mask);
    return std::make_optional<Frame>(m_frame_index++, image, features);
}

void Slam::initialize()
{
    // First first frames
    auto maybe_result = init::find_initializing_frames([this]() { return process_next_frame(); },
                                                       m_camera,
                                                       *m_feature_extractor);
    if (!maybe_result) {
        std::cout << "Initialization failed" << std::endl;
        return;
    }

    auto result = std::move(*maybe_result);
    auto ref_frame = std::make_shared<Frame>(std::move(result.ref_frame));
    auto query_frame = std::make_shared<Frame>(std::move(result.query_frame));
    std::cout << "Initializing frames: " << ref_frame->index() << " and " << query_frame->index()
              << std::endl;

    // Triangulate points
    auto matches = m_feature_extractor->match_features(ref_frame->features(),
                                                       query_frame->features());
    std::cout << "Number of matches: " << matches.size() << std::endl;
    auto points = triangulation::triangulate_points(*ref_frame, *query_frame, matches, m_camera);

    // Add points to map
    for (int i = 0; i < points.size(); i++) {
        auto match = matches[points[i].match_index];
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
    m_last_frame = query_frame;
}

void Slam::step()
{
    std::cout << "----------------------------------------" << std::endl;
    auto maybe_frame = process_next_frame();
    if (!maybe_frame) {
        std::cout << "No frame to process" << std::endl;
        return;
    }

    auto frame = std::make_shared<Frame>(std::move(*maybe_frame));
    auto last_key_frame = m_key_frames.back();

    // Initial pose estimation
    time_it("Initial pose estimation", [&]() { initial_pose_estimate(*frame); });

    // Match with map from last key frame
    time_it("Match with last key frame", [&]() { match_with_last_key_frame(*frame); });
    time_it("Optimize pose", [&]() { optimize_pose(*frame); });

    // Match with all map points
    time_it("Match with map", [&]() { match_with_map(*frame); });
    time_it("Optimize pose", [&]() { optimize_pose(*frame); });

    // Create key frame if needed
    time_it("Create key frame", [&]() {
        if (frame->num_map_matches() < 0.9 * last_key_frame->num_map_matches()) {
            std::cout << "Too few map matches, adding key frame" << std::endl;
            init_key_frame(*frame);
            m_key_frames.push_back(frame);
        }
    });

    m_last_frame = frame;
}

void Slam::initial_pose_estimate(Frame& frame)
{
    if (m_config.essential_matrix_estimation || m_key_frames.size() < 2) {
        auto pose_estimate = pose::estimate_pose(
            m_last_frame->features(),
            frame.features(),
            m_feature_extractor->match_features(m_last_frame->features(), frame.features()),
            m_camera);
        frame.set_pose(pose_estimate.pose * m_last_frame->pose());
    } else {
        frame.set_pose(m_last_frame->pose());
    }
}

void Slam::match_with_last_key_frame(Frame& frame)
{
    auto last_frame = m_key_frames.back();
    auto map_matches = m_feature_extractor->match_features(
        frame,
        m_camera,
        m_map,
        [&](const MapPoint& point) { return point.is_observed_by(last_frame.get()); });
    for (const auto& match : map_matches) {
        frame.add_map_match(match);
    }
    std::cout << "Map matches with last frame: " << map_matches.size() << std::endl;
}

void Slam::match_with_map(Frame& frame)
{
    auto map_matches = m_feature_extractor->match_features(
        frame,
        m_camera,
        m_map,
        [&](const MapPoint& point) { return true; });
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
    auto config = optimization::OptimizationConfig{
        .optimize_points = false,
        .frames = {{true, &frame}},
    };
    optimization::optimize(config, m_camera, m_map);
}

void Slam::init_key_frame(Frame& frame)
{
    // Add map associations
    auto last_key_frame = m_key_frames.back();
    for (const auto& match : frame.map_matches()) {
        m_map.add_association(frame, match);
    }

    // Triangulate unmatched points
    if (m_config.triangulate_points) {
        auto feature_matches = m_feature_extractor->match_features(last_key_frame->features(),
                                                                   frame.features());
        auto unmatched = features::unmatched_features(*last_key_frame, frame, feature_matches);
        auto points = triangulation::triangulate_points(*last_key_frame,
                                                        frame,
                                                        unmatched,
                                                        m_camera);
        for (int i = 0; i < points.size(); i++) {
            auto match = unmatched[points[i].match_index];
            m_map.create_point(points[i].position, *last_key_frame, frame, match);
        }
        std::cout << "Number of triangulated points: " << points.size() << std::endl;
    }

    // Bundle adjustment
    if (m_config.bundle_adjust) {
        std::vector<optimization::FrameConfig> frame_configs;
        for (const auto& frame : m_key_frames) {
            frame_configs.push_back({false, frame.get()});
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

} // namespace slam