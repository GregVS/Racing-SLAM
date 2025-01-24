#include <lightglue/lightglue.h>
#include "DeepFeatureExtractor.h"

namespace slam::features {

ExtractedFeatures DeepFeatureExtractor::extract_features(const cv::Mat& image, const cv::Mat& mask) const
{
    auto [keypoints, descriptors] = lightglue::FeatureExtractor(1000, 0.0005).extract_features(image);

    // Filter using mask
    std::vector<cv::KeyPoint> filtered_keypoints;
    cv::Mat filtered_descriptors;
    for (size_t i = 0; i < keypoints.size(); i++) {
        if (mask.empty() || mask.at<uchar>(keypoints[i].pt) != 0) {
            filtered_keypoints.push_back(keypoints[i]);
            filtered_descriptors.push_back(descriptors.row(i));
        }
    }

    return {filtered_keypoints, filtered_descriptors};
}

} // namespace slam::features
