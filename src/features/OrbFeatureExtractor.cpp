#include "OrbFeatureExtractor.h"

namespace slam::features {

ExtractedFeatures OrbFeatureExtractor::extract_features(const cv::Mat& image, const cv::Mat& mask) const
{
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    // Feature extraction and description
    cv::Ptr<cv::Feature2D> extractor = cv::GFTTDetector::create(3000, 0.005, 7);
    extractor->detect(gray_image, keypoints, mask);

    for (auto& keypoint : keypoints) {
        keypoint.size = 31;
    }

    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    detector->compute(gray_image, keypoints, descriptors);

    return {keypoints, descriptors};
}

} // namespace slam::features
