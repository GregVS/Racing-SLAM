#pragma once

#include <opencv2/opencv.hpp>

namespace slam {

class Camera {
  public:
    Camera(float focal_len, int width, int height);

    cv::Point3f to_camera_coordinates(const cv::Point3f &point, const cv::Mat &pose) const;
    cv::Point2f to_image_coordinates(const cv::Point3f &point, const cv::Mat &pose) const;
    cv::Point2f normalize(const cv::Point2f &point) const;

    cv::Mat get_projection_matrix(const cv::Mat &pose) const;
    const cv::Mat &get_intrinsic_matrix() const;
    bool is_visible(const cv::Point2f &point) const;

    int get_width() const;
    int get_height() const;

  private:
    const cv::Mat m_K;
    const int m_width;
    const int m_height;
};

}; // namespace slam