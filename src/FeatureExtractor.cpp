#include "FeatureExtractor.h"

namespace slam {

FeatureExtractor::FeatureExtractor() {}

ExtractedFeatures FeatureExtractor::extract_features(const cv::Mat &image,
                                                     const cv::InputArray &mask) const
{
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    // Feature extraction and description
    cv::Ptr<cv::Feature2D> extractor = cv::GFTTDetector::create(1000, 0.01, 10);
    extractor->detect(gray_image, keypoints, mask);

    for (auto &keypoint : keypoints) {
        keypoint.size = 31;
    }

    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    detector->compute(gray_image, keypoints, descriptors);

    return {keypoints, descriptors};
}

std::vector<FeatureMatch> FeatureExtractor::match_features(const ExtractedFeatures &prev_features,
                                                           const ExtractedFeatures &features) const
{
    std::vector<std::vector<cv::DMatch>> matches;
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
    matcher->knnMatch(features.descriptors, prev_features.descriptors, matches, 1);

    std::vector<FeatureMatch> feature_matches;
    for (const auto &match : matches) {
        if (match.size() > 0 && match[0].distance < MAX_ORB_DISTANCE) {
            feature_matches.push_back(FeatureMatch(match[0].trainIdx, match[0].queryIdx));
        }
    }
    return feature_matches;
}

} // namespace slam