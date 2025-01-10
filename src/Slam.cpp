#include "Slam.h"

#include "Features.h"
#include "Init.h"
#include "Triangulation.h"
#include "Optimization.h"

namespace slam {

Slam::Slam(const VideoLoader& video_loader, const Camera& camera, const cv::Mat& image_mask)
    : m_video_loader(video_loader), m_camera(camera), m_static_mask(image_mask)
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

    m_key_frames.push_back(ref_frame);
    m_key_frames.push_back(query_frame);

    // Triangulate points
    auto [points1, points2] = triangulation::get_matching_points(ref_frame->features(),
                                                                 query_frame->features(),
                                                                 result.inlier_matches);
    auto points = triangulation::triangulate_points(points1,
                                                    points2,
                                                    Eigen::Matrix4f::Identity(),
                                                    result.pose,
                                                    m_camera);

    // Add points to map
    for (int i = 0; i < points.size(); i++) {
        auto point = std::make_unique<MapPoint>(points[i].position);
        point->add_observation(ref_frame.get(), result.inlier_matches[points[i].match_index].train_index);
        point->add_observation(query_frame.get(), result.inlier_matches[points[i].match_index].query_index);
        m_map.add_point(std::move(point));

        // This is mostly for visualization
        ref_frame->add_map_match(MapPointMatch{*point, result.inlier_matches[points[i].match_index].train_index});
        query_frame->add_map_match(MapPointMatch{*point, result.inlier_matches[points[i].match_index].query_index});
    }
    std::cout << "Number of triangulated points: " << points.size() << std::endl;

    m_poses.push_back(Eigen::Matrix4f::Identity());
    m_poses.push_back(result.pose);
    m_frame = query_frame;
}

void Slam::step()
{
    auto maybe_frame = process_next_frame();
    if (!maybe_frame) {
        std::cout << "No frame to process" << std::endl;
        return;
    }
    auto frame = std::make_shared<Frame>(std::move(*maybe_frame));

    // Estimate pose based on velocity
    auto velocity = m_poses.back().block<3, 1>(0, 3) - m_poses[m_poses.size() - 2].block<3, 1>(0, 3);
    auto pose_estimate = m_poses.back();
    pose_estimate.block<3, 1>(0, 3) += velocity;

    // Match features to map
    auto matches = features::match_features(*frame, m_camera, m_map, pose_estimate);
    for (const auto& match : matches) {
        frame->add_map_match(match);
    }
    std::cout << "Number of map matches: " << frame->map_matches().size() << std::endl;

    // Optimize pose
    auto optimized_pose = optimization::optimize_pose(pose_estimate, m_map, *frame, m_camera);
    m_poses.push_back(optimized_pose);

    m_frame = frame;
}

const Map& Slam::map() const { return m_map; }

const Frame& Slam::frame() const { return *m_frame; }

const std::vector<Eigen::Matrix4f>& Slam::poses() const { return m_poses; }

} // namespace slam