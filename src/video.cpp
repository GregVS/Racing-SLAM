#include "video.h"

cv::VideoCapture initializeVideo(const std::string &videoPath)
{
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        throw std::runtime_error("Error: Could not open video file!");
    }
    cv::namedWindow("Video", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video", 1280, 720);
    return cap;
}

bool nextFrame(cv::VideoCapture &cap, cv::Mat &frame)
{
    bool isSuccess = cap.read(frame);
    if (!isSuccess) {
        std::cerr << "End of video or cannot read the frame!" << std::endl;
        return false;
    }
    return true;
}

void drawMatches(const FrameData &prevFrame, const FrameData &frame, const std::vector<cv::DMatch> &matches)
{
    cv::drawKeypoints(frame.frame, frame.keypoints, frame.frame);
    for (const auto &match : matches) {
        cv::line(frame.frame, frame.keypoints[match.queryIdx].pt,
                 prevFrame.keypoints[match.trainIdx].pt, cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("Video", frame.frame);
}