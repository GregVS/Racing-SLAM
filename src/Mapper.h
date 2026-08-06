#pragma once

#include <memory>
#include <vector>

#include "Camera.h"
#include "Map.h"
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
    Mapper(const Camera& camera, const SlamConfig& config, Map& map);

    bool needs_key_frame(const Frame& frame) const;

    /** Promotes the frame, triangulates the tracks it closes, bundle adjusts and culls */
    std::shared_ptr<KeyFrame> insert(Frame&& frame,
                                     TrackStore& tracks,
                                     const Trajectory& trajectory,
                                     const Frame& last_frame,
                                     FrameDiagnostics& diagnostics);

    void adopt(const std::shared_ptr<KeyFrame>& key_frame);

    const std::vector<std::shared_ptr<KeyFrame>>& key_frames() const;

  private:
    void triangulate_tracks(KeyFrame& key_frame,
                            TrackStore& tracks,
                            const Trajectory& trajectory,
                            FrameDiagnostics& diagnostics);
    void bundle_adjust(KeyFrame& key_frame, const Frame& last_frame);
    void cull_points(FrameDiagnostics& diagnostics);

    const Camera& m_camera;
    const SlamConfig& m_config;
    Map& m_map;
    std::vector<std::shared_ptr<KeyFrame>> m_key_frames;
};

} // namespace slam
