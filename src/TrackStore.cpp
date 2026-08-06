#include "TrackStore.h"

#include "Frame.h"

namespace slam {

TrackId TrackStore::next_id()
{
    return static_cast<TrackId>(m_next_id++);
}

void TrackStore::carry_forward(const std::vector<FeatureMatch>& matches)
{
    // train_index indexes the previous frame, query_index the current one
    std::map<TrackId, Track> carried;
    std::unordered_map<size_t, TrackId> by_keypoint;
    by_keypoint.reserve(matches.size());

    for (const auto& match : matches) {
        auto existing = m_by_keypoint.find(match.train_index);
        if (existing == m_by_keypoint.end()) {
            continue;
        }
        auto track = m_tracks.find(existing->second);
        if (track == m_tracks.end()) {
            continue;
        }
        track->second.keypoint_index = match.query_index;
        by_keypoint[match.query_index] = existing->second;
        carried.emplace(existing->second, std::move(track->second));
    }

    m_tracks = std::move(carried);
    m_by_keypoint = std::move(by_keypoint);
}

void TrackStore::extend(const Frame& frame, KeyFrame* key_frame, size_t max_sightings)
{
    for (size_t i = 0; i < frame.features().keypoints.size(); i++) {
        auto existing = m_by_keypoint.find(i);
        if (existing == m_by_keypoint.end()) {
            auto id = next_id();
            m_by_keypoint.emplace(i, id);
            m_tracks.emplace(id, Track{{}, i});
            existing = m_by_keypoint.find(i);
        }

        auto& sightings = m_tracks.at(existing->second).sightings;
        if (sightings.size() < max_sightings) {
            auto pixel = frame.keypoint(i).pt;
            sightings.push_back({frame.pose(), Eigen::Vector2f(pixel.x, pixel.y), key_frame, i});
        }
    }
}

void TrackStore::erase(TrackId id)
{
    auto track = m_tracks.find(id);
    if (track == m_tracks.end()) {
        return;
    }
    m_by_keypoint.erase(track->second.keypoint_index);
    m_tracks.erase(track);
}

const std::map<TrackId, Track>& TrackStore::tracks() const
{
    return m_tracks;
}

} // namespace slam
