#include "Features.h"

#include "Camera.h"
#include "Frame.h"
#include "Map.h"

namespace slam::features {

ExtractedFeatures extract_features(const cv::Mat& image, const cv::InputArray& mask)
{
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    // Feature extraction and description
    cv::Ptr<cv::Feature2D> extractor = cv::GFTTDetector::create(1000, 0.01, 10);
    extractor->detect(gray_image, keypoints, mask);

    for (auto& keypoint : keypoints) {
        keypoint.size = 31;
    }

    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    detector->compute(gray_image, keypoints, descriptors);

    return {keypoints, descriptors};
}

std::vector<FeatureMatch> match_features(const ExtractedFeatures& prev_features,
                                         const ExtractedFeatures& features)
{
    std::vector<std::vector<cv::DMatch>> matches;
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
    matcher->knnMatch(features.descriptors, prev_features.descriptors, matches, 1);

    std::vector<FeatureMatch> feature_matches;
    for (const auto& match : matches) {
        if (match.size() > 0 && match[0].distance < MAX_ORB_DISTANCE) {
            feature_matches.push_back(FeatureMatch(match[0].trainIdx, match[0].queryIdx));
        }
    }
    return feature_matches;
}

std::vector<MapPointMatch> match_features(const Frame& frame,
                                          const Camera& camera,
                                          const Map& map,
                                          const Eigen::Matrix4f& pose)
{
    std::vector<MapPointMatch> matches;
    for (const auto& point : map) {
        // Project point into image
        auto image_point = camera.project(pose, point.position());
        if (!camera.is_in_image(image_point))
            continue;

        // Compare to features in the region
        auto feature_indices = frame.features_in_region(image_point, 15);

        // Find the closest match
        size_t best_match_index = 0;
        float best_match_distance = MAX_ORB_DISTANCE_TO_MAP;

        for (const auto& index : feature_indices) {
            const auto& descriptor = frame.descriptor(index);

            // Ensure it is within the distance threshold
            for (const auto& [obs_keyframe, obs_index] : point.observations()) {
                auto orb_dist = cv::norm(descriptor,
                                         obs_keyframe->descriptor(obs_index),
                                         cv::NORM_HAMMING);
                if (orb_dist < best_match_distance) {
                    best_match_distance = orb_dist;
                    best_match_index = index;
                }
            }
        }

        if (best_match_distance < MAX_ORB_DISTANCE_TO_MAP) {
            matches.push_back(MapPointMatch{point, best_match_index});
        }
    }

    return matches;
}

} // namespace slam::features