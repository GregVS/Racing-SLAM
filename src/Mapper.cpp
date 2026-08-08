#include "Mapper.h"

#include <unordered_set>

#include "Frame.h"
#include "Helpers.h"
#include "LocalWindow.h"
#include "Optimization.h"
#include "Slam.h"
#include "Triangulation.h"

namespace slam {

namespace {

constexpr size_t MAX_KEY_FRAME_GAP = 20;
constexpr size_t MIN_COVISIBLE_POINTS = 50;
constexpr float MIN_COVISIBLE_FRACTION = 0.7F;

constexpr size_t BA_WINDOW = MAX_KEY_FRAME_GAP; // Must be at least MAX_KEY_FRAME_GAP
constexpr float TRACK_MIN_PARALLAX_COSINE = 0.999848F;
constexpr float TRACK_MAX_REPROJECTION_ERROR = 4.0F;
constexpr float MAX_POINT_REPROJECTION_ERROR = 3.0F;

} // namespace

Mapper::Mapper(const Camera& camera, const SlamConfig& config, Map& map)
    : m_camera(camera), m_config(config), m_map(map)
{
}

size_t Mapper::covisible_points(const Frame& frame) const
{
    auto* reference = m_key_frames.back().get();
    size_t covisible = 0;
    for (const auto& match : frame.map_matches()) {
        if (match.point.is_observed_by(reference)) {
            covisible++;
        }
    }
    return covisible;
}

bool Mapper::needs_key_frame(const Frame& frame) const
{
    const auto& last_key_frame = *m_key_frames.back();
    size_t gap = frame.index() - last_key_frame.index();
    if (gap >= MAX_KEY_FRAME_GAP) {
        return true;
    }

    size_t covisible = covisible_points(frame);
    std::cout << "Covisible with last key frame: " << covisible << " of " << frame.num_map_matches() << '\n';
    return covisible < MIN_COVISIBLE_POINTS ||
           static_cast<float>(covisible) <
               MIN_COVISIBLE_FRACTION * static_cast<float>(last_key_frame.num_map_matches());
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
        bundle_adjust(*key_frame);
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

void Mapper::bundle_adjust(KeyFrame& key_frame)
{
    auto window = optimization::build_local_window(m_key_frames, key_frame, BA_WINDOW);
    auto config = optimization::OptimizationConfig{
        .optimize_points = true,
        .frames = window,
    };

    // Store poses before optimization to reproject the single-observation points excluded from optimization
    std::vector<std::pair<Frame*, Eigen::Matrix4f>> anchors;
    anchors.reserve(window.size());
    for (const auto& frame_config : window) {
        if (frame_config.optimize) {
            anchors.emplace_back(frame_config.frame, frame_config.frame->pose());
        }
    }

    time_it("Bundle adjustment", [&]() { optimization::optimize(config, m_camera, m_map); });

    // Reproject single-observation points excluded from optimization back to their relative locations
    for (const auto& [frame, before] : anchors) {
        const auto& after = frame->pose();
        Eigen::Matrix3f rotation_before = before.block<3, 3>(0, 0);
        Eigen::Vector3f translation_before = before.block<3, 1>(0, 3);
        Eigen::Matrix3f rotation_after = after.block<3, 3>(0, 0);
        Eigen::Vector3f translation_after = after.block<3, 1>(0, 3);
        for (const auto& match : frame->map_matches()) {
            if (match.point.observations().size() > 1) {
                continue;
            }
            Eigen::Vector3f in_camera = rotation_before * match.point.position() + translation_before;
            match.point.set_position(rotation_after.transpose() * (in_camera - translation_after));
        }
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
