#include "FeatureExtractor.h"

namespace slam {

FeatureExtractor::FeatureExtractor() {}

ExtractedFeatures FeatureExtractor::extract_features(const cv::Mat &image)
{
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    // Feature extraction and description
    cv::Ptr<cv::Feature2D> extractor = cv::GFTTDetector::create(500, 0.01, 20);
    extractor->detect(gray_image, keypoints);

    for (auto &keypoint : keypoints) {
        keypoint.size = 10 + keypoint.response * 800;
    }

    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    detector->compute(gray_image, keypoints, descriptors);

    return {keypoints, descriptors};
}

} // namespace slam