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

cv::Mat OrbFeatureExtractor::refresh_descriptors(const cv::Mat& image,
                                                 const ExtractedFeatures& features) const
{
    if (features.keypoints.empty() || features.descriptors.empty()) {
        return features.descriptors;
    }

    cv::Mat gray_image;
    if (image.channels() == 1) {
        gray_image = image;
    } else {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    }

    // compute() silently drops keypoints it cannot describe, so walk the returned list to
    // recover which index each fresh descriptor belongs to
    std::vector<cv::KeyPoint> kept = features.keypoints;
    for (auto& keypoint : kept) {
        keypoint.size = 31;
    }
    cv::Mat fresh;
    m_descriptor->compute(gray_image, kept, fresh);

    cv::Mat descriptors = features.descriptors.clone();
    size_t next = 0;
    for (size_t i = 0; i < features.keypoints.size() && next < kept.size(); i++) {
        if (features.keypoints[i].pt == kept[next].pt) {
            fresh.row(next).copyTo(descriptors.row(i));
            next++;
        }
    }
    return descriptors;
}

} // namespace slam::features
