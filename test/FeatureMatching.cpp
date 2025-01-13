#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <unordered_set>

#include "Features.h"
#include "Frame.h"
#include "PoseEstimation.h"
#include "TestHelpers.h"
#include "VideoLoader.h"
#include "Visualization.h"

int main(int argc, char* argv[])
{
    auto test_data = load_test_data(LIME_ROCK_RACE_VIDEO);
    auto frames = test_data.video_loader.get_all_frames();
    std::vector<Eigen::Matrix4f> poses = {Eigen::Matrix4f::Identity()};

    slam::Visualization vis;
    vis.initialize();
    vis.run_threaded();

    std::unique_ptr<slam::Frame> prev_frame = nullptr;
    for (int i = 0; i < frames.size(); i++) {
        cv::Mat image = frames[i];
        auto features = slam::features::extract_features(image, test_data.static_mask);
        std::unique_ptr<slam::Frame> frame = std::make_unique<slam::Frame>(i, image, features);

        if (!prev_frame) {
            frame->set_pose(Eigen::Matrix4f::Identity());
            prev_frame = std::move(frame);
            continue;
        }

        // Match features
        auto matches = slam::features::match_features(prev_frame->features(), frame->features());
        auto pose_estimate = slam::pose::estimate_pose(prev_frame->features(),
                                                       frame->features(),
                                                       matches,
                                                       test_data.camera);
        frame->set_pose(pose_estimate.pose * prev_frame->pose());
        poses.push_back(pose_estimate.pose * prev_frame->pose());

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
        cv::Mat display_frame = frame->image().clone();
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
                     prev_frame->features().keypoints.at(match.train_index).pt,
                     frame->features().keypoints.at(match.query_index).pt,
                     cv::Scalar(20, 20, 20),
                     1);
        }

        vis.set_camera_poses(poses);
        vis.set_image(display_frame);
        prev_frame = std::move(frame);
    }

    return 0;
}