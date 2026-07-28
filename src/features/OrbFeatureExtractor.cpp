#include "OrbFeatureExtractor.h"

namespace slam::features {

ExtractedFeatures OrbFeatureExtractor::extract_features(const cv::Mat& image, const cv::Mat& mask) const
{
    cv::Mat gray_image;
    if (image.channels() == 1) {
        gray_image = image;
    } else {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    // Feature extraction and description
    m_detector->detect(gray_image, keypoints, mask);

    for (auto& keypoint : keypoints) {
        keypoint.size = 31;
    }

    m_descriptor->compute(gray_image, keypoints, descriptors);

    return {keypoints, descriptors};
}

} // namespace slam::features
