#include "Mapper.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "Frame.h"
#include "Helpers.h"
#include "LocalWindow.h"
#include "MapPoint.h"
#include "MotionModel.h"
#include "Optimization.h"
#include "Slam.h"
#include "Triangulation.h"
#include "features/FeatureExtractor.h"

namespace slam {

namespace {

constexpr size_t MAX_KEY_FRAME_GAP = 20;
constexpr size_t MIN_COVISIBLE_POINTS = 50;
constexpr float MIN_COVISIBLE_FRACTION = 0.7F;

// Key frame insertion criteria based on unmapped features
constexpr size_t NEW_TRACKS_THRESHOLD = 200;
constexpr size_t MIN_SIGHTINGS_FOR_TRACK = 3;
constexpr float MIN_TRACK_TRAVEL_PIXELS = 20.0F;

constexpr size_t BA_WINDOW = MAX_KEY_FRAME_GAP;        // Must be at least MAX_KEY_FRAME_GAP
constexpr size_t LOOP_COVISIBLE_KFS = BA_WINDOW;
constexpr size_t MIN_LOOP_COVISIBLE = 15;
constexpr float TRACK_MIN_PARALLAX_COSINE = 0.999848F; // 1 degree

constexpr float ROTATION_PARALLAX_FACTOR = 0.20F;    // Requires higher parallax for higher rotation
constexpr size_t MIN_NEW_POINTS_PER_KEY_FRAME = 100; // Min quota that allows accepting points with less parallax
constexpr float ANY_PARALLAX_COSINE = 1.0F;
constexpr float TRACK_MAX_REPROJECTION_ERROR = 4.0F;
constexpr float MAX_POINT_REPROJECTION_ERROR = 3.0F;

std::vector<KeyFrame*> loop_covisibles(KeyFrame& candidate)
{
    std::unordered_map<KeyFrame*, size_t> shared;
    for (const auto& match : candidate.map_matches()) {
        for (const auto& [observer, _] : match.point.observations()) {
            if (observer != &candidate) {
                shared[observer]++;
            }
        }
    }
    std::vector<std::pair<size_t, KeyFrame*>> ranked;
    ranked.reserve(shared.size());
    for (const auto& [key_frame, count] : shared) {
        if (count >= MIN_LOOP_COVISIBLE) {
            ranked.push_back({count, key_frame});
        }
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<KeyFrame*> result;
    result.push_back(&candidate);
    for (size_t i = 0; i < ranked.size() && i < LOOP_COVISIBLE_KFS; i++) {
        result.push_back(ranked[i].second);
    }
    return result;
}

std::unordered_set<MapPoint*> loop_points(const std::vector<KeyFrame*>& key_frames)
{
    std::unordered_set<MapPoint*> points;
    for (KeyFrame* key_frame : key_frames) {
        for (const auto& match : key_frame->map_matches()) {
            points.insert(&match.point);
        }
    }
    return points;
}

} // namespace

Mapper::Mapper(const Camera& camera,
               const SlamConfig& config,
               Map& map,
               const optimization::InertialInput& inertial,
               const features::BaseFeatureExtractor& extractor)
    : m_camera(camera), m_config(config), m_map(map), m_inertial(inertial),
      m_map_matcher(camera, extractor.max_distance(), extractor.norm_type())
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

size_t Mapper::unmapped_tracks(const Frame& frame, const TrackStore& tracks) const
{
    size_t count = 0;
    for (const auto& [id, track] : tracks.tracks()) {
        if (track.sightings.size() < MIN_SIGHTINGS_FOR_TRACK) {
            continue;
        }
        if (track.keypoint_index < frame.features().keypoints.size() && frame.is_matched(track.keypoint_index)) {
            continue;
        }
        const float travel = (track.sightings.back().pixel - track.sightings.front().pixel).norm();
        if (travel < MIN_TRACK_TRAVEL_PIXELS) {
            continue;
        }
        count++;
    }
    return count;
}

bool Mapper::needs_key_frame(const Frame& frame, const TrackStore& tracks) const
{
    const auto& last_key_frame = *m_key_frames.back();
    size_t gap = frame.index() - last_key_frame.index();
    if (gap >= MAX_KEY_FRAME_GAP) {
        return true;
    }

    size_t covisible = covisible_points(frame);
    const size_t waiting = unmapped_tracks(frame, tracks);
    std::cout << "Covisible with last key frame: " << covisible << " of " << frame.num_map_matches()
              << ", unmapped tracks ready " << waiting << '\n';
    if (waiting >= NEW_TRACKS_THRESHOLD) {
        return true;
    }
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

std::shared_ptr<KeyFrame>
Mapper::insert(Frame&& frame, TrackStore& tracks, const Trajectory& trajectory, FrameDiagnostics& diagnostics)
{
    auto key_frame = std::make_shared<KeyFrame>(std::move(frame));

    for (const auto& match : key_frame->map_matches()) {
        m_map.associate(*key_frame, match.point, match.keypoint_index);
    }

    if (m_config.triangulate_points) {
        time_it("Triangulate tracks", [&]() { triangulate_tracks(*key_frame, tracks, trajectory, diagnostics); });
    }
    seed_inertial_state(*key_frame);
    if (m_config.bundle_adjust) {
        bundle_adjust(*key_frame);
    }
    if (m_config.cull_points) {
        time_it("Cull points", [&]() { cull_points(diagnostics, *key_frame); });
    }

    m_key_frames.push_back(key_frame);
    return key_frame;
}

void Mapper::fuse_match(KeyFrame& frame,
                       const MapPointMatch& match,
                       const std::unordered_set<MapPoint*>& old_points)
{
    if (match.keypoint_index >= frame.features().keypoints.size()) {
        return;
    }
    MapPoint& kept = match.point;
    if (!frame.is_matched(match.keypoint_index)) {
        if (frame.is_matched(kept)) {
            return;
        }
        m_map.associate(frame, kept, match.keypoint_index);
        return;
    }
    MapPoint& discarded = frame.map_match(match.keypoint_index);
    if (&discarded == &kept || old_points.count(&discarded) != 0) {
        return;
    }
    m_map.fuse(kept, discarded);
}

void Mapper::fuse_loop(KeyFrame& query, KeyFrame& candidate, const std::vector<MapPointMatch>& inliers)
{
    auto old_frames = loop_covisibles(candidate);
    auto old_points = loop_points(old_frames);
    std::unordered_set<const KeyFrame*> old_set(old_frames.begin(), old_frames.end());

    for (const auto& inlier : inliers) {
        fuse_match(query, inlier, old_points);
    }

    std::vector<MapPoint*> sources(old_points.begin(), old_points.end());
    size_t first = m_key_frames.size() > BA_WINDOW ? m_key_frames.size() - BA_WINDOW : 0;
    for (size_t i = first; i < m_key_frames.size(); i++) {
        KeyFrame& frame = *m_key_frames[i];
        if (old_set.count(&frame) != 0) {
            continue;
        }
        auto matches = m_map_matcher.match_for_fuse(frame, sources);
        for (const auto& match : matches) {
            fuse_match(frame, match, old_points);
        }
    }
}

void Mapper::triangulate_tracks(KeyFrame& key_frame,
                                TrackStore& tracks,
                                const Trajectory& trajectory,
                                FrameDiagnostics& diagnostics)
{
    std::unordered_set<const Frame*> ba_window;
    size_t first = m_key_frames.size() > BA_WINDOW ? m_key_frames.size() - BA_WINDOW : 0;
    for (size_t i = first; i < m_key_frames.size(); i++) {
        ba_window.insert(m_key_frames[i].get());
    }

    struct Candidate {
        const Track* track;
        Eigen::Vector3f position;
        size_t keypoint_index;
        float parallax_cosine;
        float required_cosine;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(tracks.tracks().size());
    std::vector<TrackId> inconsistent;
    inconsistent.reserve(tracks.tracks().size());

    for (const auto& [track_id, track] : tracks.tracks()) {
        size_t keypoint_index = track.keypoint_index;
        if (key_frame.is_matched(keypoint_index) || track.sightings.empty()) {
            continue;
        }
        auto pixel = key_frame.keypoint(keypoint_index).pt;
        auto first_pose = trajectory.pose_at(track.sightings.front().frame_index);
        auto points = triangulation::triangulate_points({track.sightings.front().pixel},
                                                        {Eigen::Vector2f(pixel.x, pixel.y)},
                                                        first_pose,
                                                        key_frame.pose(),
                                                        m_camera,
                                                        ANY_PARALLAX_COSINE,
                                                        TRACK_MAX_REPROJECTION_ERROR);
        if (points.empty()) {
            continue; // Behind a camera, or does not reproject: a bad correspondence either way
        }

        bool consistent = true;
        for (const auto& sighting : track.sightings) {
            auto projected = m_camera.project(trajectory.pose_at(sighting.frame_index), points.front().position);
            if ((projected - sighting.pixel).norm() > TRACK_MAX_REPROJECTION_ERROR) {
                consistent = false;
                break;
            }
        }
        if (!consistent) {
            inconsistent.push_back(track_id);
            continue;
        }

        const auto& position = points.front().position;
        Eigen::Vector3f to_first = (motion::camera_center(first_pose) - position).normalized();
        Eigen::Vector3f to_now = (key_frame.camera_center() - position).normalized();

        Eigen::Matrix3f turn = key_frame.pose().block<3, 3>(0, 0) * first_pose.block<3, 3>(0, 0).transpose();
        float turned = std::acos(std::min(1.0F, std::max(-1.0F, (turn.trace() - 1.0F) / 2.0F)));

        candidates.push_back({&track,
                              position,
                              keypoint_index,
                              to_first.dot(to_now),
                              std::min(TRACK_MIN_PARALLAX_COSINE, std::cos(ROTATION_PARALLAX_FACTOR * turned))});
    }

    // Everything above threshold, then best of the rest until the quota is met
    std::vector<size_t> accepted;
    std::vector<size_t> rejected;
    accepted.reserve(candidates.size());
    for (size_t i = 0; i < candidates.size(); i++) {
        (candidates[i].parallax_cosine <= candidates[i].required_cosine ? accepted : rejected).push_back(i);
    }
    size_t topped_up = 0;
    if (accepted.size() < MIN_NEW_POINTS_PER_KEY_FRAME && !rejected.empty()) {
        std::sort(rejected.begin(), rejected.end(), [&](size_t a, size_t b) {
            return candidates[a].parallax_cosine < candidates[b].parallax_cosine;
        });
        topped_up = std::min(MIN_NEW_POINTS_PER_KEY_FRAME - accepted.size(), rejected.size());
        accepted.insert(accepted.end(), rejected.begin(), rejected.begin() + topped_up);
    }

    size_t created = 0;
    size_t consistent = 0;
    size_t observations = 0;
    for (size_t index : accepted) {
        const auto& candidate = candidates[index];
        auto& point = m_map.create_point(candidate.position, key_frame, candidate.keypoint_index);
        for (const auto& sighting : candidate.track->sightings) {
            auto* observer = sighting.key_frame;
            // Only associate with key frames in the BA window, to keep the covisible graph small
            if (observer == nullptr || observer == &key_frame || ba_window.find(observer) == ba_window.end()) {
                continue;
            }
            if (observer->is_matched(sighting.keypoint_index) || observer->is_matched(point)) {
                continue;
            }
            m_map.associate(*observer, point, sighting.keypoint_index);
            observations++;
        }

        if (candidate.track->sightings.size() >= 3) {
            point.set_track_consistent();
            consistent++;
        }
        created++;
    }

    for (const auto track_id : inconsistent) {
        tracks.erase(track_id);
    }
    diagnostics.triangulated = created;
    diagnostics.track_consistent = consistent;
    diagnostics.poisoned = inconsistent.size();
    std::cout << "Triangulated from tracks: " << created << " of " << tracks.tracks().size() << " tracks, consistent "
              << consistent << ", inconsistent " << inconsistent.size() << ", observations " << observations
              << ", topped up " << topped_up << '\n';
}

void Mapper::seed_inertial_state(KeyFrame& key_frame) const
{
    if (!m_inertial.usable() || m_key_frames.empty()) {
        return;
    }
    const KeyFrame& previous = *m_key_frames.back();
    const double from = m_inertial.time_of(previous.index());
    const double to = m_inertial.time_of(key_frame.index());
    const std::vector<imu::Sample> samples = m_inertial.stream->between(from, to);
    if (samples.size() < 2) {
        return;
    }

    const imu::Preintegrated summary = imu::preintegrate(samples, m_inertial.noise, previous.inertial().bias);
    InertialState state;
    state.velocity = imu::predict(inertial_state(previous), summary, m_inertial.gravity).velocity;
    state.bias = previous.inertial().bias;
    key_frame.set_inertial(state);
}

void Mapper::bundle_adjust(KeyFrame& key_frame, bool fix_oldest)
{
    auto window = optimization::build_local_window(m_key_frames, key_frame, BA_WINDOW, fix_oldest);

    // Store poses before optimization to reproject the single-observation points excluded from optimization
    std::vector<std::pair<Frame*, Eigen::Matrix4f>> anchors;
    anchors.reserve(window.size());
    for (const auto& frame_config : window) {
        if (frame_config.optimize) {
            anchors.emplace_back(frame_config.frame, frame_config.frame->pose());
        }
    }

    time_it("Bundle adjustment", [&]() { optimization::bundle_adjust(window, m_camera, m_map, m_inertial); });

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

void Mapper::cull_points(FrameDiagnostics& diagnostics, KeyFrame& key_frame)
{
    std::unordered_set<MapPoint*> local;
    auto include_points = [&](const Frame& frame) {
        for (const auto& match : frame.map_matches()) {
            local.insert(&match.point);
        }
    };
    size_t first = m_key_frames.size() > BA_WINDOW ? m_key_frames.size() - BA_WINDOW : 0;
    for (size_t i = first; i < m_key_frames.size(); i++) {
        include_points(*m_key_frames[i]);
    }
    include_points(key_frame);

    std::vector<MapPoint*> points_to_remove;
    for (MapPoint* point : local) {
        float error = 0.0;
        size_t num_projected = 0;
        for (const auto& [frame, index] : point->observations()) {
            auto projected = m_camera.project(frame->pose(), point->position());
            auto image_point = Eigen::Vector2f(frame->keypoint(index).pt.x, frame->keypoint(index).pt.y);
            error += (projected - image_point).norm();
            num_projected++;
        }
        if (num_projected > 0 && error / static_cast<float>(num_projected) > MAX_POINT_REPROJECTION_ERROR) {
            points_to_remove.push_back(point);
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
