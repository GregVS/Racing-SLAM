#include <iostream>
#include <opencv2/opencv.hpp>

#include "FeatureExtractor.h"
#include "VideoLoader.h"

int main(int argc, char *argv[])
{
    std::string video_path = "videos/lime-rock-race.mp4";

    slam::VideoLoader video_loader(video_path);
    slam::FeatureExtractor feature_extractor;

    cv::Mat mask = cv::Mat::zeros(video_loader.get_height(), video_loader.get_width(), CV_8UC1);
    mask(cv::Rect(0, 0, video_loader.get_width(), video_loader.get_height() * 0.7)) = 255;

    while (true) {
        cv::Mat frame = video_loader.get_next_frame();
        if (frame.empty()) {
            break;
        }

        // Feature extraction
        auto start = std::chrono::high_resolution_clock::now();
        auto features = feature_extractor.extract_features(frame, mask);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Feature extraction took " << duration.count() * 1000 << " ms" << std::endl;

        // Draw keypoints
        cv::Mat frame_with_keypoints;
        cv::drawKeypoints(frame,
                          features.keypoints,
                          frame_with_keypoints,
                          cv::Scalar(0, 255, 0),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow("Feature Extraction", frame_with_keypoints);

        cv::waitKey(1);
    }

    cv::destroyAllWindows();

    return 0;
}