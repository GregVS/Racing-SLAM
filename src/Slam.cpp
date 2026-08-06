#include "Slam.h"

#include "Frame.h"
#include "Helpers.h"
#include "Initialization.h"
#include "features/FeatureExtractor.h"

namespace slam {

namespace {

constexpr size_t MAX_TRACK_SIGHTINGS = 100;

} // namespace

Slam::Slam(const VideoLoader& video_loader,
           const Camera& camera,
           const cv::Mat& image_mask,
           std::unique_ptr<features::BaseFeatureExtractor> feature_extractor,
           const SlamConfig& config)
    : m_video_loader(video_loader), m_camera(camera), m_static_mask(image_mask),
      m_feature_extractor(std::move(feature_extractor)), m_config(config),
      m_tracker(m_camera, m_static_mask, *m_feature_extractor, m_config, m_map), m_mapper(m_camera, m_config, m_map)
{
}

void Slam::initialize()
{
    auto result = initialize_map(m_video_loader, m_camera, m_static_mask, *m_feature_extractor, m_config, m_map);
    if (!result.ref_frame || !result.query_frame) {
        return;
    }

    m_frame_index = result.next_frame_index;
    m_mapper.adopt(result.ref_frame);
    m_mapper.adopt(result.query_frame);
    m_tracker.set_last_frame(result.query_frame);

    record_pose(*result.ref_frame);
    record_pose(*result.query_frame);
}

bool Slam::step()
{
    std::cout << "---------------------------------------- frame " << m_frame_index << "\n";
    m_diagnostics = FrameDiagnostics{};
    auto image = m_video_loader.get_next_frame();
    if (image.empty()) {
        std::cout << "No frame to process\n";
        return false;
    }

    // Recompute pose relative to the reference key frame
    if (m_tracker.has_last_frame() && m_tracker.last_frame().index() < m_trajectory.size()) {
        auto& last_frame = m_tracker.last_frame();
        last_frame.set_pose(m_trajectory.pose_at(last_frame.index()));
    }

    auto last_key_frame = m_mapper.key_frames().back();
    auto frame = m_tracker.track(
        image, m_frame_index++, m_trajectory, *last_key_frame, m_mapper.key_frames().size());

    std::shared_ptr<KeyFrame> promoted;
    time_it("Create key frame", [&]() {
        if (m_mapper.needs_key_frame(*frame)) {
            std::cout << "Adding key frame after " << frame->index() - last_key_frame->index() << " frames\n";
            promoted = m_mapper.insert(std::move(*frame), m_tracker.tracks(), m_tracker.last_frame(), m_diagnostics);
            frame = promoted;
        }
    });

    m_tracker.set_last_frame(frame);
    record_pose(*frame);
    m_diagnostics.map_size = m_map.size();

    m_tracker.tracks().extend(*frame, promoted.get(), MAX_TRACK_SIGHTINGS);
    return true;
}

void Slam::record_pose(const Frame& frame)
{
    const auto& key_frames = m_mapper.key_frames();
    m_trajectory.record(frame, key_frames.empty() ? nullptr : key_frames.back().get());
}

float Slam::reprojection_error() const
{
    float error = 0.0;
    int num_projected = 0;
    for (const auto& frame : m_mapper.key_frames()) {
        for (const auto& match : frame->map_matches()) {
            const auto& point = match.point;
            auto projected = m_camera.project(frame->pose(), point.position());
            auto image_point =
                Eigen::Vector2f(frame->keypoint(match.keypoint_index).pt.x, frame->keypoint(match.keypoint_index).pt.y);
            error += (projected - image_point).stableNorm();
            num_projected++;
        }
    }
    return error / num_projected;
}

const FrameDiagnostics& Slam::diagnostics() const
{
    return m_diagnostics;
}

const Map& Slam::map() const
{
    return m_map;
}

const Frame& Slam::frame() const
{
    return m_tracker.last_frame();
}

std::vector<Eigen::Matrix4f> Slam::poses() const
{
    std::vector<Eigen::Matrix4f> poses;
    for (const auto& frame : m_mapper.key_frames()) {
        poses.push_back(frame->pose());
    }
    return poses;
}

std::vector<Eigen::Matrix4f> Slam::trajectory() const
{
    return m_trajectory.poses();
}

} // namespace slam
