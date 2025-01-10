#include "Slam.h"

#include "Features.h"
#include "Init.h"
#include "Triangulation.h"

namespace slam {

Slam::Slam(const VideoLoader& video_loader, const Camera& camera, const cv::Mat& image_mask)
    : m_video_loader(video_loader), m_camera(camera), m_static_mask(image_mask)
{
}

std::shared_ptr<Frame> Slam::process_next_frame()
{
    auto image = m_video_loader.get_next_frame();
    auto features = features::extract_features(image, m_static_mask);
    return std::make_shared<Frame>(m_frame_index++, image, features);
}

void Slam::initialize()
{
    // First first frames
    auto result = init::find_initializing_frames([this]() { return process_next_frame(); },
                                                 m_camera);
    if (!result.ref_frame || !result.query_frame) {
        std::cout << "Initialization failed" << std::endl;
        return;
    }

    // Create the first two keyframes
    auto keyframe1 = std::make_shared<KeyFrame>(result.ref_frame->index(),
                                                Eigen::Matrix4f::Identity(),
                                                result.ref_frame->features());
    auto keyframe2 = std::make_shared<KeyFrame>(result.query_frame->index(),
                                                result.pose,
                                                result.query_frame->features());
    m_key_frames.push_back(keyframe1);
    m_key_frames.push_back(keyframe2);

    // Triangulate points
    auto [points1, points2] = triangulation::get_matching_points(keyframe1->features(),
                                                                 keyframe2->features(),
                                                                 result.inlier_matches);
    auto points = triangulation::triangulate_points(points1,
                                                    points2,
                                                    keyframe1->pose(),
                                                    keyframe2->pose(),
                                                    m_camera);

    // Add points to map
    for (int i = 0; i < points.size(); i++) {
        auto point = std::make_unique<MapPoint>(points[i].position);
        point->add_observation(keyframe1, result.inlier_matches[points[i].match_index].train_index);
        point->add_observation(keyframe2, result.inlier_matches[points[i].match_index].query_index);
        m_map.add_point(std::move(point));
    }

    m_pose = result.pose;
}

void Slam::step()
{
    auto frame = process_next_frame();

    // Match features to map
    auto matches = features::match_features(*frame, m_camera, m_map, m_pose);
    for (const auto& match : matches) {
        frame->add_map_match(match);
    }

    m_frame = std::move(frame);
}

const std::vector<std::shared_ptr<KeyFrame>>& Slam::key_frames() const { return m_key_frames; }

const Map& Slam::map() const { return m_map; }

const std::shared_ptr<Frame>& Slam::frame() const { return m_frame; }

} // namespace slam