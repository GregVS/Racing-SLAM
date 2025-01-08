#pragma once

#include <opencv2/opencv.hpp>

namespace slam {

class Camera {
  public:
    Camera(float focal_len, int width, int height);

    const cv::Mat &get_intrinsic_matrix() const;

    int get_width() const;
    int get_height() const;

  private:
    const cv::Mat m_K;
    const int m_width;
    const int m_height;
};

}; // namespace slam