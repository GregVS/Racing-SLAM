#include "Slam.h"

#include <unordered_set>

#include "Frame.h"
#include "Helpers.h"
#include "MotionModel.h"
#include "Optimization.h"
#include "Triangulation.h"

namespace slam {

namespace {

constexpr size_t BA_WINDOW = 10;
constexpr size_t MAX_KEY_FRAME_GAP = 20;
constexpr size_t MIN_TRACKED_POINTS = 50;
constexpr float TRACK_MIN_PARALLAX_COSINE = 0.999848F;
constexpr float TRACK_MAX_REPROJECTION_ERROR = 4.0F;

} // namespace

bool Slam::needs_key_frame(const Frame& frame, const Frame& last_key_frame) const
{
    if (frame.num_map_matches() < MIN_TRACKED_POINTS) {
        return true;
    }

    size_t gap = frame.index() - last_key_frame.index();
    return gap >= MAX_KEY_FRAME_GAP ||
           static_cast<float>(frame.num_map_matches()) < 0.9F * static_cast<float>(last_key_frame.num_map_matches());
}

void Slam::triangulate_tracks(Frame& frame)
{
    size_t created = 0;
    size_t validated = 0;
    std::vector<size_t> inconsistent;
    inconsistent.reserve(m_tracks.size());
    for (const auto& [keypoint_index, track] : m_tracks) {
        if (frame.is_matched(keypoint_index) || track.observations.empty()) {
            continue;
        }
        auto pixel = frame.keypoint(keypoint_index).pt;
        auto points = triangulation::triangulate_points({track.observations.front().second},
                                                        {Eigen::Vector2f(pixel.x, pixel.y)},
                                                        track.observations.front().first,
                                                        frame.pose(),
                                                        m_camera,
                                                        TRACK_MIN_PARALLAX_COSINE,
                                                        TRACK_MAX_REPROJECTION_ERROR);
        if (points.empty()) {
            continue;
        }

        // Check if reprojects consistently into all observed frames
        bool consistent = true;
        for (const auto& [pose, observed] : track.observations) {
            if ((m_camera.project(pose, points.front().position) - observed).norm() > TRACK_MAX_REPROJECTION_ERROR) {
                consistent = false;
                break;
            }
        }
        if (!consistent) {
            inconsistent.push_back(keypoint_index);
            continue;
        }

        auto& point = m_map.create_point(points.front().position, frame, keypoint_index);
        if (track.observations.size() >= 3) {
            point.set_track_consistent();
            validated++;
        }
        created++;
    }

    for (const auto keypoint_index : inconsistent) {
        m_tracks.erase(keypoint_index);
    }
    std::cout << "Triangulated from tracks: " << created << " of " << m_tracks.size() << " tracks, consistent "
              << validated << ", poisoned " << inconsistent.size() << '\n';
}

void Slam::init_key_frame(Frame& frame)
{
    // Add map associations
    for (const auto& match : frame.map_matches()) {
        m_map.add_association(frame, match);
    }

    // Triangulate from feature tracks rather than key frame to key frame matches
    if (m_config.triangulate_points) {
        time_it("Triangulate tracks", [&]() { triangulate_tracks(frame); });
    }

    // Local bundle adjustment (covisbility graph)
    if (m_config.bundle_adjust) {
        size_t first_optimized = m_key_frames.size() > BA_WINDOW ? m_key_frames.size() - BA_WINDOW : 2;
        std::unordered_set<const Frame*> window{&frame};
        for (size_t i = first_optimized; i < m_key_frames.size(); i++) {
            window.insert(m_key_frames[i].get());
        }

        // Add any frames that share map point observations with the window
        std::unordered_set<const Frame*> anchors;
        for (const Frame* window_frame : window) {
            for (const auto& match : window_frame->map_matches()) {
                for (const auto& [observer, _] : match.point.observations()) {
                    if (window.find(observer) == window.end()) {
                        anchors.insert(observer);
                    }
                }
            }
        }

        // Build optimization config
        std::vector<optimization::FrameConfig> frame_configs;
        frame_configs.reserve(m_key_frames.size() + 1);
        for (const auto& key_frame : m_key_frames) {
            if (window.find(key_frame.get()) != window.end()) {
                // Historical poses define the accepted motion chain. Moving them without a
                // pose graph would leave adjacent non-keyframes in stale coordinate frames.
                frame_configs.push_back({false, key_frame.get()});
            } else if (anchors.find(key_frame.get()) != anchors.end()) {
                frame_configs.push_back({false, key_frame.get()});
            }
        }
        frame_configs.push_back({m_config.metric_steps.empty(), &frame, m_config.metric_steps.empty()});

        // Step constraints
        std::vector<optimization::StepConstraint> step_constraints;
        step_constraints.reserve(frame_configs.size());
        if (!m_config.metric_steps.empty()) {
            std::unordered_set<const Frame*> selected;
            for (const auto& frame_config : frame_configs) {
                selected.insert(frame_config.frame);
            }
            for (size_t i = 1; i < m_key_frames.size(); i++) {
                const Frame* previous = m_key_frames[i - 1].get();
                const Frame* current = m_key_frames[i].get();
                if (selected.count(previous) && selected.count(current)) {
                    step_constraints.push_back(
                        {previous, current, metric_distance(previous->index(), current->index())});
                }
            }
            step_constraints.push_back(
                {m_key_frames.back().get(), &frame, metric_distance(m_key_frames.back()->index(), frame.index())});
        }

        auto config = optimization::OptimizationConfig{
            .optimize_points = true,
            .frames = frame_configs,
            .step_constraints = step_constraints,
        };
        std::vector<std::pair<Frame*, Eigen::Matrix4f>> original_poses;
        original_poses.reserve(frame_configs.size());
        for (const auto& frame_config : frame_configs) {
            if (frame_config.optimize) {
                original_poses.push_back({frame_config.frame, frame_config.frame->pose()});
            }
        }
        std::vector<std::pair<MapPoint*, Eigen::Vector3f>> original_points;
        for (auto& point : m_map) {
            original_points.push_back({&point, point.position()});
        }

        bool optimized = false;
        time_it("Bundle adjustment", [&]() { optimized = optimization::optimize(config, m_camera, m_map); });
        bool healthy = optimized;
        if (healthy && !m_config.metric_steps.empty() &&
            !motion::is_rotation_plausible(m_last_frame->pose(), frame.pose(), m_config.seconds_per_frame)) {
            healthy = false;
        }
        if (healthy && !m_config.metric_steps.empty()) {
            for (const auto& constraint : step_constraints) {
                float elapsed =
                    static_cast<float>(constraint.b->index() - constraint.a->index()) * m_config.seconds_per_frame;
                if (!motion::is_rotation_plausible(constraint.a->pose(), constraint.b->pose(), elapsed)) {
                    healthy = false;
                    break;
                }
            }
        }
        if (!healthy && optimized) {
            for (const auto& [snapshot_frame, pose] : original_poses) {
                snapshot_frame->set_pose(pose);
            }
            for (const auto& [point, position] : original_points) {
                point->set_position(position);
            }
            std::cout << "Local bundle adjustment rolled back by motion health contract\n";
        }
    }

    // Cull points
    if (m_config.cull_points) {
        time_it("Cull points", [&]() { cull_points(); });
    }
}

float Slam::metric_distance(size_t from, size_t to) const
{
    float distance = 0;
    for (size_t i = from; i < to && i < m_config.metric_steps.size(); i++) {
        distance += m_config.metric_steps[i];
    }
    return distance;
}

void Slam::cull_points()
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
        if (num_projected > 0 && error / static_cast<float>(num_projected) > 3.0F) {
            points_to_remove.push_back(&point);
        }
    }

    std::cout << "Number of points to remove: " << points_to_remove.size() << '\n';
    for (const auto& point : points_to_remove) {
        m_map.remove_point(point);
    }
}

} // namespace slam
