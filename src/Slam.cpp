#include "Slam.h"

#include "Features.h"
#include "Init.h"
#include "Optimization.h"
#include "PoseEstimation.h"
#include "Triangulation.h"

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
    std::cout << "Initializing frames: " << ref_frame->index() << " and " << query_frame->index()
              << std::endl;

    // Triangulate points
    auto points = triangulation::triangulate_points(*ref_frame,
                                                    *query_frame,
                                                    result.matches,
                                                    m_camera);

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

    auto pose_estimate = pose::estimate_pose(
        last_frame->features(),
        frame->features(),
        features::match_features(last_frame->features(), frame->features()),
        m_camera);
    frame->set_pose(pose_estimate.pose * last_frame->pose());

    // Match with map
    auto matches = features::match_features(*frame, m_camera, m_map);
    for (const auto& match : matches) {
        frame->add_map_match(match);
        const_cast<MapPoint&>(match.point).add_observation(frame.get(), match.keypoint_index);
    }
    std::cout << "Number of map matches: " << matches.size() << std::endl;

    // Add frame to key frames
    m_key_frames.push_back(frame);
    m_frame = frame;
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
            error += (projected - image_point).norm();
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