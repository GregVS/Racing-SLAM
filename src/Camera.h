#pragma once

#include <opencv2/opencv.hpp>

namespace slam
{

class Camera {
public:
    Camera(cv::Mat K, int width, int height);

    cv::Point3f to_camera_coordinates(const cv::Point3f &point, const cv::Mat &pose) const;
    cv::Point2f to_image_coordinates(const cv::Point3f &point, const cv::Mat &pose) const;

    cv::Mat get_projection_matrix(const cv::Mat &pose) const;
    const cv::Mat &get_intrinsic_matrix() const;

    int get_width() const;
    int get_height() const;

private:
    const cv::Mat m_K;
    const int m_width;
    const int m_height;
};

bool within_frame(const cv::Point2f &point, const Camera &camera);

};