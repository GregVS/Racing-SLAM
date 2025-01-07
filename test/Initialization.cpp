#include <iostream>
#include <opencv2/opencv.hpp>
#include <unordered_set>

#include "FeatureExtractor.h"
#include "Initializer.h"
#include "VideoLoader.h"

/**
 * Finds two frames that will be used to initialize the system and displays them along with their
 * keypoint matches.
 */
int main(int argc, char* argv[])
{
    std::string video_path = "videos/lime-rock-race.mp4";

    slam::VideoLoader video_loader(video_path);
    slam::FeatureExtractor feature_extractor;
    slam::Camera camera(914, video_loader.get_width(), video_loader.get_height());
    slam::Initializer initializer(camera);

    cv::Mat mask = cv::Mat::zeros(video_loader.get_height(), video_loader.get_width(), CV_8UC1);
    mask(cv::Rect(0, 0, video_loader.get_width(), video_loader.get_height() * 0.7)) = 255;

    std::shared_ptr<slam::Frame> ref_frame;
    std::shared_ptr<slam::Frame> query_frame;
    slam::InitializerResult result;

    // Find the initializing frames
    auto frames = video_loader.get_all_frames();
    for (int i = 0; i < frames.size(); i++) {
        cv::Mat curr_image = frames[i];
        auto features = feature_extractor.extract_features(curr_image, mask);

        auto frame = std::make_shared<slam::Frame>(i, curr_image);
        frame->set_features(features);

        auto result_opt = initializer.try_initialize(frame);
        if (result_opt) {
            ref_frame = initializer.ref_frame();
            query_frame = frame;
            result = result_opt.value();
            break;
        }

        std::cout << "Failed to initialize with frame " << i << std::endl;
    }

    std::cout << "Ref frame: " << ref_frame->index() << std::endl;
    std::cout << "Query frame: " << query_frame->index() << std::endl;
    std::cout << "Pose: " << result.pose.inverse() << std::endl;

    // Cycle through the two frames (press any key to cycle)
    auto selected_frame = ref_frame;
    while (true) {
        selected_frame = selected_frame == ref_frame ? query_frame : ref_frame;
        std::cout << "Selected frame: " << selected_frame->index() << std::endl;

        cv::Mat display_frame = selected_frame->image().clone();
        std::vector<cv::KeyPoint> matched_keypoints;
        for (int i = 0; i < result.inlier_matches.size(); i++) {
            int query_index = result.inlier_matches[i].query_index;
            int train_index = result.inlier_matches[i].train_index;
            if (selected_frame == query_frame) {
                matched_keypoints.push_back(query_frame->features().keypoints.at(query_index));
            } else {
                matched_keypoints.push_back(ref_frame->features().keypoints.at(train_index));
            }
        }
        cv::drawKeypoints(display_frame,
                          matched_keypoints,
                          display_frame,
                          cv::Scalar(0, 255, 0),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        for (int i = 0; i < result.inlier_matches.size(); i++) {
            cv::line(display_frame,
                     ref_frame->features().keypoints.at(result.inlier_matches[i].train_index).pt,
                     query_frame->features().keypoints.at(result.inlier_matches[i].query_index).pt,
                     cv::Scalar(20, 20, 20),
                     1);
        }
        cv::imshow("frame", display_frame);
        cv::waitKey(0);
    }

    cv::destroyAllWindows();
    return 0;
}