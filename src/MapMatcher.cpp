#include "MapMatcher.h"

#include "Camera.h"
#include "Frame.h"
#include "Map.h"

namespace slam {

namespace {

constexpr float SEARCH_RADIUS = 20.0F;
constexpr float MIN_VIEWING_ANGLE_COSINE = 0.5F;

// Used to filter out matches that are too close or too far away
constexpr float MAX_NEARER_RATIO = 2.0F;
constexpr float MAX_FURTHER_RATIO = 1.25F;
constexpr float MATCH_RATIO = 0.75F;

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
    struct ProposedMatch {
        MapPoint* point;
        float dist;
        size_t keypoint_index;
    };

    std::vector<ProposedMatch> proposed_matches(frame.features().keypoints.size());
    for (size_t i = 0; i < frame.features().keypoints.size(); ++i) {
        proposed_matches[i] = {.point = nullptr, .dist = m_max_descriptor_distance, .keypoint_index = 0};
    }

    for (auto& point : map) {
        if ((required_observer != nullptr && !point.is_observed_by(required_observer)) || frame.is_matched(point)) {
            continue;
        }

        auto image_point = m_camera.project(frame.pose(), point.position());
        if (!m_camera.is_in_image(image_point)) {
            continue;
        }

        auto ray = point.position() - frame.camera_center();
        auto viewing_normal = point.avg_viewing_normal();
        if (viewing_normal.dot(ray.normalized()) < MIN_VIEWING_ANGLE_COSINE) {
            continue;
        }

        // Only match points that were observed from a similar distance
        auto [nearest, furthest] = point.observed_distance_range();
        float distance = ray.norm();
        if (distance < nearest / MAX_NEARER_RATIO || distance > furthest * MAX_FURTHER_RATIO) {
            continue;
        }

        auto feature_indices = frame.features_in_region(image_point, SEARCH_RADIUS);

        size_t best_match_index = 0;
        float best_match_distance = m_max_descriptor_distance;

        for (const auto& index : feature_indices) {
            // Claimed by an earlier pass
            if (frame.is_matched(index)) {
                continue;
            }

            const auto descriptor = frame.descriptor(index);

            for (const auto& [obs_keyframe, obs_index] : point.observations()) {
                auto orb_dist = cv::norm(descriptor, obs_keyframe->descriptor(obs_index), m_norm_type);
                if (orb_dist < best_match_distance) {
                    best_match_distance = orb_dist;
                    best_match_index = index;
                }
            }
        }

        if (best_match_distance < proposed_matches[best_match_index].dist) {
            proposed_matches[best_match_index] = {
                .point = &point, .dist = best_match_distance, .keypoint_index = best_match_index};
        }
    }

    std::vector<MapPointMatch> final_matches;
    for (const auto& proposed_match : proposed_matches) {
        if (proposed_match.point != nullptr) {
            final_matches.push_back(MapPointMatch{*proposed_match.point, proposed_match.keypoint_index});
        }
    }
    return final_matches;
}

} // namespace slam
