#pragma once

#include <opencv2/opencv.hpp>

namespace slam {

class VideoLoader {
  public:
    VideoLoader(const std::string &video_path);

    cv::Mat get_next_frame();

    int get_width() const;
    int get_height() const;

    int get_fps() const;

  private:
    cv::VideoCapture m_video_capture;
};

}; // namespace slam