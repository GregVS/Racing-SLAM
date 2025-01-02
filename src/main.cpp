#include <chrono>

#include "Features.h"
#include "VideoDisplay.h"
#include "Graphics.h"
#include "MapMatching.h"
#include "Map.h"
#include "Frame.h"
#include "BundleAdjustment.h"

int main()
{
    std::string videoPath = "videos/highway.mp4";

    try {
        cv::VideoCapture cap = slam::init_video(videoPath);

        slam::Graphics graphics(800, 600);

        int W = 1920;
        int H = 1080;
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        K.at<double>(0, 0) = 800;
        K.at<double>(1, 1) = 800;
        K.at<double>(0, 2) = W / 2;
        K.at<double>(1, 2) = H / 2;

        slam::Camera camera{ K, W, H };

        slam::Map map(camera);

        auto start = std::chrono::high_resolution_clock::now();

        bool run_video = true;

        while (graphics.is_running()) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            if (duration.count() > 200 && run_video) {
                start = end;
                cv::Mat image = slam::next_frame(cap);

                // Process frame
                if (map.get_frames().empty()) {
                    slam::Frame &frame = map.add_frame(slam::extract_features(image, map.get_next_frame_id()));
                    frame.set_pose(cv::Mat::eye(4, 4, CV_64F));
                    continue;
                }

                slam::Frame &prevFrame = map.get_last_frame();
                slam::Frame &frame = map.add_frame(slam::extract_features(image, map.get_next_frame_id()));

                auto matches = slam::match_features(prevFrame, frame);
                auto poseEstimate = slam::estimate_pose(prevFrame, frame, camera, matches);
                slam::draw_matches(prevFrame, frame, poseEstimate.filteredMatches);

                frame.set_pose(poseEstimate.pose);

                slam::piggyback_prev_frame_matches(map, prevFrame, frame, poseEstimate.filteredMatches);
                slam::match_map_points(map, frame);
                slam::triangulate_points(map, prevFrame, frame, poseEstimate.filteredMatches);

                if (map.get_frames().size() > 2) {
                    slam::BundleAdjustment ba;
                    ba.optimize_map(map);
                    slam::cull_points(map);
                }
                if (map.get_frames().size() > 25) {
                    run_video = false;
                }

                std::cout << "Map reprojection error: " << slam::reprojection_error(map) << std::endl;

                std::cout << "Number of map points: " << map.get_map_points().size() << std::endl;

                cv::waitKey(1);
            }

            // Draw 3d scene
            graphics.draw_scene(map);
        }

        cap.release();
        cv::destroyAllWindows();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
