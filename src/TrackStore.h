#pragma once

#include <Eigen/Dense>
#include <map>
#include <unordered_map>
#include <vector>

#include "features/FeatureExtractor.h"

namespace slam {

class Frame;
class KeyFrame;

enum class TrackId : size_t {};

struct TrackSighting {
    Eigen::Matrix4f pose;
    Eigen::Vector2f pixel;
    KeyFrame* key_frame = nullptr;
    size_t keypoint_index = 0;
};

struct Track {
    std::vector<TrackSighting> sightings;
    size_t keypoint_index = 0; // in the current frame
};

/** Feature tracks through the current frame. A track keeps its id; its keypoint index changes */
class TrackStore {
  public:
    void carry_forward(const std::vector<FeatureMatch>& matches);
    void extend(const Frame& frame, KeyFrame* key_frame, size_t max_sightings);

    void erase(TrackId id);

    const std::map<TrackId, Track>& tracks() const;

  private:
    TrackId next_id();

    std::map<TrackId, Track> m_tracks;
    std::unordered_map<size_t, TrackId> m_by_keypoint;
    size_t m_next_id = 0;
};

} // namespace slam
