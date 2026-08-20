#include "MapMatcher.h"

#include "Camera.h"
#include "Frame.h"
#include "Map.h"
#include "MapPoint.h"

namespace slam {

namespace {

constexpr float SEARCH_RADIUS = 20.0F;
constexpr float MIN_VIEWING_ANGLE_COSINE = 0.5F;

// Used to filter out matches that are too close or too far away
constexpr float MAX_NEARER_RATIO = 2.0F;
constexpr float MAX_FURTHER_RATIO = 1.25F;
constexpr float MATCH_RATIO = 0.75F;

struct ProposedMatch {
    MapPoint* point;
    float dist;
};

std::vector<ProposedMatch> empty_proposals(size_t count, float max_descriptor_distance)
{
    std::vector<ProposedMatch> proposed(count);
    for (auto& proposal : proposed) {
        proposal = {.point = nullptr, .dist = max_descriptor_distance};
    }
    return proposed;
}

std::vector<MapPointMatch> accepted_matches(const std::vector<ProposedMatch>& proposed)
{
    std::vector<MapPointMatch> matches;
    for (size_t i = 0; i < proposed.size(); i++) {
        if (proposed[i].point != nullptr) {
            matches.push_back(MapPointMatch{*proposed[i].point, i});
        }
    }
    return matches;
}

void match_via_reproject(const Camera& camera,
                         float max_descriptor_distance,
                         cv::NormTypes norm_type,
                         const Frame& frame,
                         MapPoint& point,
                         std::vector<ProposedMatch>& proposed,
                         bool replace)
{
    if (frame.is_matched(point)) {
        return;
    }

    auto image_point = camera.project(frame.pose(), point.position());
    if (!camera.is_in_image(image_point)) {
        return;
    }

    auto ray = point.position() - frame.camera_center();
    auto viewing_normal = point.avg_viewing_normal();
    if (viewing_normal.dot(ray.normalized()) < MIN_VIEWING_ANGLE_COSINE) {
        return;
    }

    // Only match points that were observed from a similar distance
    auto [nearest, furthest] = point.observed_distance_range();
    float distance = ray.norm();
    if (distance < nearest / MAX_NEARER_RATIO || distance > furthest * MAX_FURTHER_RATIO) {
        return;
    }

    auto feature_indices = frame.features_in_region(image_point, SEARCH_RADIUS);

    size_t best_match_index = 0;
    float best_match_distance = max_descriptor_distance;

    for (const auto& index : feature_indices) {
        if (!replace && frame.is_matched(index)) {
            continue;
        }

        const auto descriptor = frame.descriptor(index);
        for (const auto& [obs_keyframe, obs_index] : point.observations()) {
            auto orb_dist = cv::norm(descriptor, obs_keyframe->descriptor(obs_index), norm_type);
            if (orb_dist < best_match_distance) {
                best_match_distance = orb_dist;
                best_match_index = index;
            }
        }
    }

    if (best_match_distance < proposed[best_match_index].dist) {
        proposed[best_match_index] = {.point = &point, .dist = best_match_distance};
    }
}

} // namespace

MapMatcher::MapMatcher(const Camera& camera, float max_descriptor_distance, cv::NormTypes norm_type)
    : m_camera(camera), m_max_descriptor_distance(max_descriptor_distance), m_norm_type(norm_type)
{
}

std::vector<MapPointMatch> MapMatcher::match_map(const Frame& frame, Map& map) const
{
    return match(frame, map, nullptr);
}

std::vector<MapPointMatch> MapMatcher::match_key_frame(const Frame& frame, Map& map, KeyFrame* key_frame) const
{
    return match(frame, map, key_frame);
}

std::vector<MapPointMatch> MapMatcher::match_for_fuse(const Frame& frame, const std::vector<MapPoint*>& points) const
{
    auto proposed = empty_proposals(frame.features().keypoints.size(), m_max_descriptor_distance);
    for (MapPoint* point : points) {
        if (point == nullptr) {
            continue;
        }
        match_via_reproject(m_camera, m_max_descriptor_distance, m_norm_type, frame, *point, proposed, true);
    }
    return accepted_matches(proposed);
}

std::vector<MapPointMatch> MapMatcher::match_descriptors(const Frame& frame, const KeyFrame& key_frame) const
{
    std::vector<MapPoint*> points;
    std::vector<cv::Mat> descriptor_list;
    points.reserve(key_frame.num_map_matches());
    descriptor_list.reserve(key_frame.num_map_matches());
    for (const auto& match : key_frame.map_matches()) {
        points.push_back(&match.point);
        descriptor_list.push_back(key_frame.descriptor(match.keypoint_index));
    }
    if (points.empty() || frame.features().descriptors.empty()) {
        return {};
    }

    cv::Mat map_descriptors;
    cv::vconcat(descriptor_list, map_descriptors);
    const int k = map_descriptors.rows >= 2 ? 2 : 1;
    std::vector<std::vector<cv::DMatch>> knn;
    cv::BFMatcher matcher(m_norm_type);
    matcher.knnMatch(frame.features().descriptors, map_descriptors, knn, k);

    std::vector<MapPointMatch> matches;
    for (const auto& candidates : knn) {
        if (candidates.empty() || candidates[0].distance > m_max_descriptor_distance) {
            continue;
        }
        // Lowes ratio test
        if (k > 1 && candidates.size() > 1 && candidates[0].distance > MATCH_RATIO * candidates[1].distance) {
            continue;
        }
        matches.push_back(
            {*points[static_cast<size_t>(candidates[0].trainIdx)], static_cast<size_t>(candidates[0].queryIdx)});
    }
    return matches;
}

std::vector<MapPointMatch> MapMatcher::match(const Frame& frame, Map& map, KeyFrame* required_observer) const
{
    auto proposed = empty_proposals(frame.features().keypoints.size(), m_max_descriptor_distance);
    for (auto& point : map) {
        if (required_observer != nullptr && !point.is_observed_by(required_observer)) {
            continue;
        }
        match_via_reproject(m_camera, m_max_descriptor_distance, m_norm_type, frame, point, proposed, false);
    }
    return accepted_matches(proposed);
}

} // namespace slam
