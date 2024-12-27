#include "video.h"

cv::VideoCapture initializeVideo(const std::string &videoPath)
{
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        throw std::runtime_error("Error: Could not open video file!");
    }
    return cap;
}

bool displayFrame(cv::VideoCapture &cap)
{
    cv::Mat frame;
    bool isSuccess = cap.read(frame);
    if (!isSuccess) {
        std::cerr << "End of video or cannot read the frame!" << std::endl;
        return false;
    }
    cv::imshow("Video Frame", frame);
    return true;
}

void playVideo(cv::VideoCapture &cap)
{
    while (displayFrame(cap)) {
        cv::waitKey(1);
    }
}