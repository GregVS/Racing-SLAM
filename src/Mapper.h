#pragma once

#include <memory>
#include <vector>

#include "Camera.h"
#include "Map.h"
#include "Optimization.h"
#include "TrackStore.h"
#include "Trajectory.h"

namespace slam {

class Frame;
class KeyFrame;
struct SlamConfig;
struct FrameDiagnostics;

/** Owns the key frames and everything that grows the map: promotion, triangulation, local bundle
 * adjustment and point culling */
class Mapper {
  public:
    Mapper(const Camera& camera, const SlamConfig& config, Map& map, const optimization::InertialInput& inertial);

    bool needs_key_frame(const Frame& frame, const TrackStore& tracks) const;

    /** Promotes the frame, triangulates the tracks it closes, bundle adjusts and culls */
    std::shared_ptr<KeyFrame>
    insert(Frame&& frame, TrackStore& tracks, const Trajectory& trajectory, FrameDiagnostics& diagnostics);

    void adopt(const std::shared_ptr<KeyFrame>& key_frame);

    const std::vector<std::shared_ptr<KeyFrame>>& key_frames() const;

  private:
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
    void bundle_adjust(KeyFrame& key_frame);
    void cull_points(FrameDiagnostics& diagnostics, KeyFrame& key_frame);

    const Camera& m_camera;
    const SlamConfig& m_config;
    Map& m_map;
    const optimization::InertialInput& m_inertial;
    std::vector<std::shared_ptr<KeyFrame>> m_key_frames;
};

} // namespace slam
