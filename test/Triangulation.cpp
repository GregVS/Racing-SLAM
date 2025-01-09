#include <iostream>
#include <opencv2/opencv.hpp>
#include <unordered_set>

#include "Triangulation.h"
#include "TestHelpers.h"
#include "Visualization.h"

/**
 * Runs through the video and triangulates points between every two frames.
 */
int main(int argc, char* argv[])
{
    auto test_data = load_test_data();

    slam::Visualization visualization("Triangulation");
    visualization.initialize();
    std::thread visualization_thread([&]() { visualization.run(); });

    std::shared_ptr<slam::Frame> ref_frame;
    std::shared_ptr<slam::Frame> query_frame;

    std::shared_ptr<slam::Frame> prev_frame;
    std::vector<Eigen::Matrix4f> poses = {Eigen::Matrix4f::Identity()};
    std::vector<Eigen::Vector3f> points_3d;

    auto images = test_data.video_loader.get_all_frames();
    for (int i = 0; i < images.size(); i++) {
        cv::Mat curr_image = images[i];
        auto curr_frame = std::make_shared<slam::Frame>(i, curr_image);
        curr_frame->set_features(
            test_data.feature_extractor.extract_features(curr_image, test_data.static_mask));

        if (!prev_frame) {
            prev_frame = curr_frame;
            continue;
        }

        auto matches = test_data.feature_extractor.match_features(prev_frame->features(),
                                                                  curr_frame->features());
        auto pose_estimate = test_data.pose_estimator.estimate_pose(matches,
                                                                    prev_frame->features(),
                                                                    curr_frame->features(),
                                                                    test_data.camera);
        auto global_pose = pose_estimate.pose * poses.back();
        auto [points1, points2] = slam::get_matching_points(prev_frame->features(),
                                                            curr_frame->features(),
                                                            pose_estimate.inlier_matches);
        auto points_3d_new =
            slam::triangulate_points(points1, points2, poses.back(), global_pose, test_data.camera);
        points_3d.insert(points_3d.end(), points_3d_new.begin(), points_3d_new.end());
        poses.push_back(global_pose);

        std::cout << "Points 3D: " << points_3d_new.size() << std::endl;

        // Update visualization
        visualization.set_camera_poses(poses);
        visualization.set_points(points_3d);
        visualization.set_image(curr_image);

        prev_frame = curr_frame;
    }

    visualization_thread.join();
    cv::destroyAllWindows();
    return 0;
}