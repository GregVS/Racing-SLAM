#include <iostream>
#include <opencv2/opencv.hpp>
#include <unordered_set>

#include "FeatureExtractor.h"
#include "PoseEstimator.h"
#include "VideoLoader.h"

int main(int argc, char* argv[])
{
    std::string video_path = "videos/lime-rock-race.mp4";

    slam::VideoLoader video_loader(video_path);
    slam::FeatureExtractor feature_extractor;
    slam::PoseEstimator pose_estimator;
    slam::Camera camera(914, video_loader.get_width(), video_loader.get_height());

    cv::Mat mask = cv::Mat::zeros(video_loader.get_height(), video_loader.get_width(), CV_8UC1);
    mask(cv::Rect(0, 0, video_loader.get_width(), video_loader.get_height() * 0.7)) = 255;

    auto frames = video_loader.get_all_frames();

    for (int i = 1; i < frames.size(); i++) {
        cv::Mat prev_frame = frames[i - 1];
        cv::Mat frame = frames[i];

        auto prev_features = feature_extractor.extract_features(prev_frame, mask);
        auto features = feature_extractor.extract_features(frame, mask);

        auto matches = feature_extractor.match_features(prev_features, features);
        auto pose_estimate = pose_estimator.estimate_pose(matches, prev_features, features, camera);

        std::vector<cv::KeyPoint> inliner_keypoints;
        std::vector<cv::KeyPoint> outlier_keypoints;
        std::vector<cv::KeyPoint> unmatched_keypoints;

        {
            std::unordered_set<int> matched_indices; // includes outliers
            std::unordered_set<int> inliner_indices;
            for (const auto& match : matches) {
                matched_indices.insert(match.query_index);
            }
            for (int i = 0; i < matches.size(); i++) {
                if (pose_estimate.inliers[i] == 1) {
                    inliner_indices.insert(matches[i].query_index);
                }
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
        std::cout << "Pose: " << pose_estimate.pose.inverse() << std::endl;

        // Draw the matched keypoints in green and the unmatched keypoints in red
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
        for (int i = 0; i < matches.size(); i++) {
            if (pose_estimate.inliers[i] == 0) continue;

            cv::line(display_frame,
                     prev_features.keypoints.at(matches[i].train_index).pt,
                     features.keypoints.at(matches[i].query_index).pt,
                     cv::Scalar(20, 20, 20),
                     1);
        }
        cv::imshow("frame", display_frame);
        cv::waitKey(0);
    }

    cv::destroyAllWindows();

    return 0;
}