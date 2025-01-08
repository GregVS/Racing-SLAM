#include "Camera.h"

namespace slam {

static cv::Mat intrinsic_from_focal_len(float focal_len, int width, int height)
{
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = focal_len;
    K.at<double>(1, 1) = focal_len;
    K.at<double>(0, 2) = width / 2;
    K.at<double>(1, 2) = height / 2;
    return K;
}

Camera::Camera(float focal_len, int width, int height)
    : m_K(intrinsic_from_focal_len(focal_len, width, height)), m_width(width), m_height(height)
{
}

int Camera::get_width() const { return m_width; }

int Camera::get_height() const { return m_height; }

const cv::Mat& Camera::get_intrinsic_matrix() const { return m_K; }

}; // namespace slam