#pragma once

#include <memory>
#include <unordered_set>
#include <vector>

#include "Camera.h"
#include "Map.h"
#include "MapMatcher.h"
#include "Optimization.h"
#include "TrackStore.h"
#include "Trajectory.h"

namespace slam {

class Frame;
class KeyFrame;
struct SlamConfig;
struct FrameDiagnostics;

namespace features {
class BaseFeatureExtractor;
}

/** Owns the key frames and everything that grows the map: promotion, triangulation, local bundle
 * adjustment and point culling */
class Mapper {
  public:
    Mapper(const Camera& camera,
           const SlamConfig& config,
           Map& map,
           const optimization::InertialInput& inertial,
           const features::BaseFeatureExtractor& extractor);

    bool needs_key_frame(const Frame& frame, const TrackStore& tracks) const;

    /** Promotes the frame, triangulates the tracks it closes, bundle adjusts and culls */
    std::shared_ptr<KeyFrame>
    insert(Frame&& frame, TrackStore& tracks, const Trajectory& trajectory, FrameDiagnostics& diagnostics);

    void adopt(const std::shared_ptr<KeyFrame>& key_frame);

    const std::vector<std::shared_ptr<KeyFrame>>& key_frames() const;

    void fuse_loop(KeyFrame& query, KeyFrame& candidate, const std::vector<MapPointMatch>& inliers);
    void bundle_adjust(KeyFrame& key_frame, bool fix_oldest = false);

  private:
    void fuse_match(KeyFrame& frame,
                    const MapPointMatch& match,
                    const std::unordered_set<MapPoint*>& old_points);
    /** Numbers of covisible points with the last key frame */
    size_t covisible_points(const Frame& frame) const;

    /** Tracks that are good quality but not yet triangulated */
    size_t unmapped_tracks(const Frame& frame, const TrackStore& tracks) const;

    void triangulate_tracks(KeyFrame& key_frame,
                            TrackStore& tracks,
                            const Trajectory& trajectory,
                            FrameDiagnostics& diagnostics);

    /** Initial inertial state estimate for new key frame */
    void seed_inertial_state(KeyFrame& key_frame) const;
    void cull_points(FrameDiagnostics& diagnostics, KeyFrame& key_frame);

    const Camera& m_camera;
    const SlamConfig& m_config;
    Map& m_map;
    const optimization::InertialInput& m_inertial;
    MapMatcher m_map_matcher;
    std::vector<std::shared_ptr<KeyFrame>> m_key_frames;
};

} // namespace slam
