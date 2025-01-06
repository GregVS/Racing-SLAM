#include <iostream>
#include <opencv2/opencv.hpp>

#include "FeatureExtractor.h"
#include "VideoLoader.h"

int main(int argc, char *argv[])
{
    std::string video_path;
    if (argc > 1) {
        video_path = argv[1];
    } else {
        std::cerr << "Usage: " << argv[0] << " <video_path>" << std::endl;
        return 1;
    }

    slam::VideoLoader video_loader(video_path);
    slam::FeatureExtractor feature_extractor;

    while (true) {
        cv::Mat frame = video_loader.get_next_frame();
        if (frame.empty()) {
            break;
        }

        // Feature extraction
        auto start = std::chrono::high_resolution_clock::now();
        auto features = feature_extractor.extract_features(frame);
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
        cv::imshow("frame", frame_with_keypoints);

        cv::waitKey(1);
    }

    cv::destroyAllWindows();

    return 0;
}