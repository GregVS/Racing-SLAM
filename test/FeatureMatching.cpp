#include <iostream>
#include <opencv2/opencv.hpp>
#include <unordered_set>

#include "Features.h"
#include "PoseEstimation.h"
#include "TestHelpers.h"
#include "VideoLoader.h"

int main(int argc, char* argv[])
{
    auto test_data = load_test_data();

    auto frames = test_data.video_loader.get_all_frames();

    for (int i = 1; i < frames.size(); i++) {
        cv::Mat prev_frame = frames[i - 1];
        cv::Mat frame = frames[i];

        // Extract features
        auto prev_features = slam::features::extract_features(prev_frame, test_data.static_mask);
        auto features = slam::features::extract_features(frame, test_data.static_mask);

        // Match features
        auto matches = slam::features::match_features(prev_features, features);
        auto pose_estimate = slam::pose::estimate_pose(matches,
                                                       prev_features,
                                                       features,
                                                       test_data.camera);

        // Group keypoints into inliers, outliers, and unmatched
        std::vector<cv::KeyPoint> inliner_keypoints;
        std::vector<cv::KeyPoint> outlier_keypoints;
        std::vector<cv::KeyPoint> unmatched_keypoints;
        {
            std::unordered_set<int> matched_indices; // includes outliers
            std::unordered_set<int> inliner_indices;
            for (const auto& match : matches) {
                matched_indices.insert(match.query_index);
            }
            for (const auto& match : pose_estimate.inlier_matches) {
                inliner_indices.insert(match.query_index);
            }

            for (int i = 0; i < features.keypoints.size(); i++) {
                if (inliner_indices.find(i) != inliner_indices.end()) {
                    inliner_keypoints.push_back(features.keypoints[i]);
                } else if (matched_indices.find(i) != matched_indices.end()) {
                    outlier_keypoints.push_back(features.keypoints[i]);
                } else {
                    unmatched_keypoints.push_back(features.keypoints[i]);
                }
            }
        }

        std::cout << "Inliers: " << inliner_keypoints.size()
                  << " Outliers: " << outlier_keypoints.size()
                  << " Unmatched: " << unmatched_keypoints.size() << std::endl;

        // Draw the keypoints
        cv::Mat display_frame = frame.clone();
        cv::drawKeypoints(display_frame,
                          inliner_keypoints,
                          display_frame,
                          cv::Scalar(0, 255, 0),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::drawKeypoints(display_frame,
                          outlier_keypoints,
                          display_frame,
                          cv::Scalar(255, 0, 0),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::drawKeypoints(display_frame,
                          unmatched_keypoints,
                          display_frame,
                          cv::Scalar(0, 0, 255),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // Draw a line to the previous frame
        for (const auto& match : pose_estimate.inlier_matches) {
            cv::line(display_frame,
                     prev_features.keypoints.at(match.train_index).pt,
                     features.keypoints.at(match.query_index).pt,
                     cv::Scalar(20, 20, 20),
                     1);
        }
        cv::imshow("frame", display_frame);
        cv::waitKey(1);
    }

    cv::destroyAllWindows();

    return 0;
}