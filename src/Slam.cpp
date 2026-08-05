#include "Slam.h"

#include "Frame.h"
#include "Helpers.h"
#include "features/FeatureExtractor.h"

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
    time_it("Initial pose estimation", [&]() { inlier_matches = initial_pose_estimate(*frame, tracked); });
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
            std::cout << "Adding key frame after " << frame->index() - last_key_frame->index() << " frames"
                      << std::endl;
            init_key_frame(*frame);
            m_key_frames.push_back(frame);
        }
    });

    m_last_frame = frame;
    record_pose(*frame);

    // Extend tracks
    for (size_t i = 0; i < frame->features().keypoints.size(); i++) {
        auto& observations = m_tracks[i].observations;
        if (observations.size() < 100) {
            auto pixel = frame->keypoint(i).pt;
            observations.push_back({frame->pose(), Eigen::Vector2f(pixel.x, pixel.y)});
        }
    }
    return true;
}

void Slam::record_pose(const Frame& frame)
{
    // Frames dropped during initialization keep the previous pose so that the trajectory stays
    // indexed by frame index
    auto fill = m_trajectory.empty() ? Eigen::Matrix4f::Identity() : m_trajectory.back();
    m_trajectory.resize(frame.index() + 1, fill);
    m_trajectory[frame.index()] = frame.pose();
}

float Slam::reprojection_error() const
{
    float error = 0.0;
    int num_projected = 0;
    for (const auto& frame : m_key_frames) {
        for (const auto& match : frame->map_matches()) {
            auto point = match.point;
            auto projected = m_camera.project(frame->pose(), point.position());
            auto image_point =
                Eigen::Vector2f(frame->keypoint(match.keypoint_index).pt.x, frame->keypoint(match.keypoint_index).pt.y);
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
