#include "VideoDisplay.h"

namespace slam
{

cv::VideoCapture init_video(const std::string &videoPath)
{
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        throw std::runtime_error("Error: Could not open video file!");
    }
    cv::namedWindow("Video", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video", 1280, 720);
    return cap;
}

cv::Mat next_frame(cv::VideoCapture &cap)
{
    cv::Mat frame;
    bool is_success = cap.read(frame);
    if (!is_success) {
        std::cerr << "End of video or cannot read the frame!" << std::endl;
        return cv::Mat();
    }
    return frame;
}

void draw_matches(const Frame &prev_frame, const Frame &frame, const std::vector<cv::DMatch> &matches)
{
    cv::Mat frame_image = frame.get_image().clone();
    for (const auto &match : matches) {
        cv::line(frame_image, frame.get_keypoint(match.queryIdx).pt, prev_frame.get_keypoint(match.trainIdx).pt,
                 cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("Video", frame_image);
}

// This is for debugging purposes
void draw_correspondences(const MapPoint &point)
{
    auto observations = point.get_observations_vector();
    auto sorted_obs = std::vector<std::pair<Frame *, int> >(observations.begin(), observations.end());
    std::sort(sorted_obs.begin(), sorted_obs.end(),
              [](const auto &a, const auto &b) { return a.first->get_id() < b.first->get_id(); });

    std::cout << "Number of observations: " << sorted_obs.size() << std::endl;
    for (auto &[frame, keypointIndex] : sorted_obs) {
        std::cout << "Frame ID: " << frame->get_id() << std::endl;
        // Draw the frame
        cv::Mat frameImage = frame->get_image().clone();
        cv::circle(frameImage, frame->get_keypoint(keypointIndex).pt, 5, cv::Scalar(0, 0, 255), -1);

        cv::imshow("Frame", frameImage);
        cv::waitKey(0);
    }
    std::cout << "--------------------------------" << std::endl;
}

};