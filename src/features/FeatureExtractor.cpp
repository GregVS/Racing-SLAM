#include "FeatureExtractor.h"

#include "../Camera.h"
#include "../Frame.h"
#include "../Map.h"

namespace slam::features {

cv::Mat BaseFeatureExtractor::refresh_descriptors(const cv::Mat& image,
                                                  const ExtractedFeatures& features) const
{
    return features.descriptors;
}

std::vector<FeatureMatch>
BaseFeatureExtractor::match_features(const ExtractedFeatures& prev_features,
                                     const ExtractedFeatures& features) const
{
    std::vector<std::vector<cv::DMatch>> matches;
    auto matcher = cv::BFMatcher::create(norm_type(), true);
    matcher->knnMatch(features.descriptors, prev_features.descriptors, matches, 1);

    std::vector<FeatureMatch> feature_matches;
    for (const auto& match : matches) {
        if (match.size() > 0 && match[0].distance < max_distance()) {
            feature_matches.push_back(FeatureMatch(match[0].trainIdx, match[0].queryIdx));
        }
    }
    return feature_matches;
}

std::vector<FeatureMatch> unmatched_features(const Frame& frame1,
                                             const Frame& frame2,
                                             const std::vector<FeatureMatch>& matches)
{
    std::vector<FeatureMatch> unmatched;
    for (const auto& match : matches) {
        // train_index = frame1, query_index = frame2
        if (!frame1.is_matched(match.train_index) && !frame2.is_matched(match.query_index)) {
            unmatched.push_back(match);
        }
    }
    return unmatched;
}

} // namespace slam::features