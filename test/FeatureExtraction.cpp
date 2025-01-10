#include <iostream>
#include <opencv2/opencv.hpp>

#include "Features.h"
#include "TestHelpers.h"
#include "VideoLoader.h"

int main(int argc, char* argv[])
{
    auto test_data = load_test_data();

    while (true) {
        cv::Mat image = test_data.video_loader.get_next_frame();
        if (image.empty()) {
            break;
        }

        // Feature extraction
        auto features = time_it<slam::ExtractedFeatures>("Feature extraction", [&]() {
            return slam::features::extract_features(image, test_data.static_mask);
        });

        std::cout << "Features: " << features.keypoints.size() << std::endl;

        // Draw keypoints
        cv::Mat frame_with_keypoints;
        cv::drawKeypoints(image,
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