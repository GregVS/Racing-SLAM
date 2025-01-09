#include <iostream>
#include <opencv2/opencv.hpp>
#include <unordered_set>

#include "Initializer.h"
#include "TestHelpers.h"
#include "Visualization.h"

/**
 * Finds two frames that will be used to initialize the system and displays them along with their
 * keypoint matches.
 */
int main(int argc, char* argv[])
{
    auto test_data = load_test_data();
    slam::Initializer initializer(test_data.camera);
    std::shared_ptr<slam::Frame> ref_frame;
    std::shared_ptr<slam::Frame> query_frame;
    slam::InitializerResult result;

    // Find the initializing frames
    auto frames = test_data.video_loader.get_all_frames();
    for (int i = 0; i < frames.size(); i++) {
        cv::Mat curr_image = frames[i];
        auto features =
            test_data.feature_extractor.extract_features(curr_image, test_data.static_mask);

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

    slam::Visualization visualization("Initialization");
    visualization.initialize();
    visualization.run_threaded();

    visualization.set_camera_poses({Eigen::Matrix4f::Identity(), result.pose});

    // Cycle through the two frames (press any key to cycle)
    auto selected_frame = ref_frame;
    while (!visualization.has_quit()) {
        selected_frame = selected_frame == ref_frame ? query_frame : ref_frame;
        std::cout << "Selected frame: " << selected_frame->index() << std::endl;

        cv::Mat display_frame = selected_frame->image().clone();
        std::vector<cv::KeyPoint> matched_keypoints;
        for (const auto& match : result.inlier_matches) {
            int query_index = match.query_index;
            int train_index = match.train_index;
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
        for (const auto& match : result.inlier_matches) {
            cv::line(display_frame,
                     ref_frame->features().keypoints.at(match.train_index).pt,
                     query_frame->features().keypoints.at(match.query_index).pt,
                     cv::Scalar(20, 20, 20),
                     1);
        }
        visualization.set_image(display_frame);
        visualization.wait_for_keypress();
    }

    return 0;
}