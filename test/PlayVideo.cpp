#include "VideoLoader.h"
#include <iostream>
#include <opencv2/opencv.hpp>

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

    while (true) {
        cv::Mat frame = video_loader.get_next_frame();
        if (frame.empty()) {
            break;
        }

        cv::imshow("frame", frame);
        cv::waitKey(1000 / video_loader.get_fps());
    }

    cv::destroyAllWindows();

    return 0;
}