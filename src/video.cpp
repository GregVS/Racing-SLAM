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

cv::Mat nextFrame(cv::VideoCapture &cap)
{
    cv::Mat frame;
    bool isSuccess = cap.read(frame);
    if (!isSuccess) {
        std::cerr << "End of video or cannot read the frame!" << std::endl;
        return cv::Mat();
    }
    return frame;
}

void drawMatches(const Frame &prevFrame, const Frame &frame, const std::vector<cv::DMatch> &matches)
{
    cv::Mat frameImage = frame.getImage().clone();
    for (const auto &match : matches) {
        cv::line(frameImage, frame.getKeypoint(match.queryIdx).pt, prevFrame.getKeypoint(match.trainIdx).pt,
                 cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("Video", frameImage);
}

void showCorrespondences(const MapPoint &point)
{
    auto observations = point.getObservations();
    auto sortedObservations = std::vector<std::pair<Frame *, int> >(observations.begin(), observations.end());
    std::sort(sortedObservations.begin(), sortedObservations.end(),
              [](const auto &a, const auto &b) { return a.first->getId() < b.first->getId(); });

    std::cout << "Number of observations: " << point.getObservations().size() << std::endl;
    for (auto &[frame, keypointIndex] : sortedObservations) {
        std::cout << "Frame ID: " << frame->getId() << std::endl;
        // Draw the frame
        cv::Mat frameImage = frame->getImage().clone();
        cv::circle(frameImage, frame->getKeypoint(keypointIndex).pt, 5, cv::Scalar(0, 0, 255), -1);

        cv::imshow("Frame", frameImage);
        cv::waitKey(0);
    }
    std::cout << "--------------------------------" << std::endl;
}