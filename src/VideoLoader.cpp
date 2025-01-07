#include "VideoLoader.h"

namespace slam {

VideoLoader::VideoLoader(const std::string &video_path) : m_video_capture(video_path) {}

cv::Mat VideoLoader::get_next_frame()
{
    cv::Mat frame;
    m_video_capture >> frame;
    return frame;
}

std::vector<cv::Mat> VideoLoader::get_all_frames()
{
    std::vector<cv::Mat> frames;
    while (true) {
        cv::Mat frame = get_next_frame();
        if (frame.empty()) {
            break;
        }
        frames.push_back(frame);
    }
    return frames;
}

int VideoLoader::get_width() const { return m_video_capture.get(cv::CAP_PROP_FRAME_WIDTH); }

int VideoLoader::get_height() const { return m_video_capture.get(cv::CAP_PROP_FRAME_HEIGHT); }

int VideoLoader::get_fps() const { return m_video_capture.get(cv::CAP_PROP_FPS); }

} // namespace slam