#include "Mapper.h"

#include <unordered_set>

#include "Frame.h"
#include "Helpers.h"
#include "LocalWindow.h"
#include "MotionModel.h"
#include "Optimization.h"
#include "Slam.h"
#include "Triangulation.h"

namespace slam {

namespace {

constexpr size_t BA_WINDOW = 10;
constexpr size_t MAX_KEY_FRAME_GAP = 20;
constexpr size_t MIN_TRACKED_POINTS = 50;
constexpr float TRACK_MIN_PARALLAX_COSINE = 0.999848F;
constexpr float TRACK_MAX_REPROJECTION_ERROR = 4.0F;
constexpr float MAX_POINT_REPROJECTION_ERROR = 3.0F;

} // namespace

Mapper::Mapper(const Camera& camera, const SlamConfig& config, Map& map)
    : m_camera(camera), m_config(config), m_map(map)
{
}

bool Mapper::needs_key_frame(const Frame& frame) const
{
    const auto& last_key_frame = *m_key_frames.back();
    if (frame.num_map_matches() < MIN_TRACKED_POINTS) {
        return true;
    }

    size_t gap = frame.index() - last_key_frame.index();
    return gap >= MAX_KEY_FRAME_GAP ||
           static_cast<float>(frame.num_map_matches()) < 0.9F * static_cast<float>(last_key_frame.num_map_matches());
}

void Mapper::adopt(const std::shared_ptr<KeyFrame>& key_frame)
{
    m_key_frames.push_back(key_frame);
}

const std::vector<std::shared_ptr<KeyFrame>>& Mapper::key_frames() const
{
    return m_key_frames;
}

std::shared_ptr<KeyFrame> Mapper::insert(Frame&& frame,
                                         TrackStore& tracks,
                                         const Trajectory& trajectory,
                                         const Frame& last_frame,
                                         FrameDiagnostics& diagnostics)
{
    auto key_frame = std::make_shared<KeyFrame>(std::move(frame));

    for (const auto& match : key_frame->map_matches()) {
        m_map.associate(*key_frame, match.point, match.keypoint_index);
    }

    if (m_config.triangulate_points) {
        time_it("Triangulate tracks", [&]() { triangulate_tracks(*key_frame, tracks, trajectory, diagnostics); });
    }
    if (m_config.bundle_adjust) {
        bundle_adjust(*key_frame, last_frame);
    }
    if (m_config.cull_points) {
        time_it("Cull points", [&]() { cull_points(diagnostics); });
    }

    m_key_frames.push_back(key_frame);
    return key_frame;
}

void Mapper::triangulate_tracks(KeyFrame& key_frame,
                                TrackStore& tracks,
                                const Trajectory& trajectory,
                                FrameDiagnostics& diagnostics)
{
    size_t created = 0;
    size_t validated = 0;
    size_t ba_frames = 0;
    std::unordered_set<const Frame*> ba_window;
    size_t first = m_key_frames.size() > BA_WINDOW ? m_key_frames.size() - BA_WINDOW : 0;
    for (size_t i = first; i < m_key_frames.size(); i++) {
        ba_window.insert(m_key_frames[i].get());
    }

    std::vector<TrackId> inconsistent;
    inconsistent.reserve(tracks.tracks().size());
    for (const auto& [track_id, track] : tracks.tracks()) {
        size_t keypoint_index = track.keypoint_index;
        if (key_frame.is_matched(keypoint_index) || track.sightings.empty()) {
            continue;
        }
        auto pixel = key_frame.keypoint(keypoint_index).pt;
        auto points = triangulation::triangulate_points({track.sightings.front().pixel},
                                                        {Eigen::Vector2f(pixel.x, pixel.y)},
                                                        trajectory.pose_at(track.sightings.front().frame_index),
                                                        key_frame.pose(),
                                                        m_camera,
                                                        TRACK_MIN_PARALLAX_COSINE,
                                                        TRACK_MAX_REPROJECTION_ERROR);
        if (points.empty()) {
            continue;
        }

        bool consistent = true;
        for (const auto& sighting : track.sightings) {
            if ((m_camera.project(trajectory.pose_at(sighting.frame_index), points.front().position) - sighting.pixel)
                    .norm() > TRACK_MAX_REPROJECTION_ERROR) {
                consistent = false;
                break;
            }
        }
        if (!consistent) {
            inconsistent.push_back(track_id);
            continue;
        }

        auto& point = m_map.create_point(points.front().position, key_frame, keypoint_index);
        for (const auto& sighting : track.sightings) {
            // Only associate with key frames in the BA window, to keep the covisible graph small
            if (sighting.key_frame == nullptr || sighting.key_frame == &key_frame ||
                ba_window.find(sighting.key_frame) == ba_window.end()) {
                continue;
            }
            auto* observer = sighting.key_frame;
            if (observer->is_matched(sighting.keypoint_index) || observer->is_matched(point)) {
                continue;
            }
            m_map.associate(*observer, point, sighting.keypoint_index);
            ba_frames++;
        }

        if (track.sightings.size() >= 3) {
            point.set_track_consistent();
            validated++;
        }
        created++;
    }

    for (const auto track_id : inconsistent) {
        tracks.erase(track_id);
    }
    diagnostics.triangulated = created;
    diagnostics.track_consistent = validated;
    diagnostics.poisoned = inconsistent.size();
    std::cout << "Triangulated from tracks: " << created << " of " << tracks.tracks().size() << " tracks, consistent "
              << validated << ", inconsistent " << inconsistent.size() << ", key frame anchors " << ba_frames << '\n';
}

void Mapper::bundle_adjust(KeyFrame& key_frame, const Frame& last_frame)
{
    auto window = optimization::build_local_window(m_key_frames, key_frame, BA_WINDOW, m_config.metric_steps);
    auto config = optimization::OptimizationConfig{
        .optimize_points = true,
        .frames = window.frames,
        .step_constraints = window.step_constraints,
    };

    optimization::Snapshot snapshot(config, m_map, true);
    bool optimized = false;
    time_it("Bundle adjustment", [&]() { optimized = optimization::optimize(config, m_camera, m_map); });

    bool healthy = optimized;
    if (healthy && !m_config.metric_steps.empty() &&
        !motion::is_rotation_plausible(last_frame.pose(), key_frame.pose(), m_config.seconds_per_frame)) {
        healthy = false;
    }
    if (healthy && !m_config.metric_steps.empty()) {
        for (const auto& constraint : window.step_constraints) {
            float elapsed = static_cast<float>(constraint.b->index() - constraint.a->index()) *
                            m_config.seconds_per_frame;
            if (!motion::is_rotation_plausible(constraint.a->pose(), constraint.b->pose(), elapsed)) {
                healthy = false;
                break;
            }
        }
    }
    if (!healthy && optimized) {
        snapshot.restore();
        std::cout << "Local bundle adjustment rolled back by motion health contract\n";
    }
}

void Mapper::cull_points(FrameDiagnostics& diagnostics)
{
    std::vector<MapPoint*> points_to_remove;
    for (auto& point : m_map) {
        float error = 0.0;
        size_t num_projected = 0;
        for (const auto& [frame, index] : point.observations()) {
            auto projected = m_camera.project(frame->pose(), point.position());
            auto image_point = Eigen::Vector2f(frame->keypoint(index).pt.x, frame->keypoint(index).pt.y);
            error += (projected - image_point).norm();
            num_projected++;
        }
        if (num_projected > 0 && error / static_cast<float>(num_projected) > MAX_POINT_REPROJECTION_ERROR) {
            points_to_remove.push_back(&point);
        }
    }

    std::cout << "Number of points to remove: " << points_to_remove.size() << '\n';
    diagnostics.culled.reserve(points_to_remove.size());
    for (const auto& point : points_to_remove) {
        diagnostics.culled.push_back(point->position());
        m_map.remove_point(point);
    }
}

} // namespace slam
