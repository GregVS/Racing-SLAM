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

cv::Mat Camera::get_projection_matrix(const cv::Mat &pose) const
{
    cv::Mat E = pose.inv();
    cv::Mat projectionMatrix = m_K * E.rowRange(0, 3);
    return projectionMatrix;
}

cv::Point3f Camera::to_camera_coordinates(const cv::Point3f &point, const cv::Mat &pose) const
{
    cv::Mat pointMat = cv::Mat(4, 1, CV_64F);
    pointMat.at<double>(0, 0) = point.x;
    pointMat.at<double>(1, 0) = point.y;
    pointMat.at<double>(2, 0) = point.z;
    pointMat.at<double>(3, 0) = 1;

    cv::Mat cameraPoint = pose * pointMat;
    cameraPoint /= cameraPoint.at<double>(3, 0);
    return cv::Point3f(cameraPoint.at<double>(0, 0),
                       cameraPoint.at<double>(1, 0),
                       cameraPoint.at<double>(2, 0));
}

cv::Point2f Camera::to_image_coordinates(const cv::Point3f &point, const cv::Mat &pose) const
{
    cv::Mat pointMat = cv::Mat(4, 1, CV_64F);
    pointMat.at<double>(0, 0) = point.x;
    pointMat.at<double>(1, 0) = point.y;
    pointMat.at<double>(2, 0) = point.z;
    pointMat.at<double>(3, 0) = 1;

    cv::Mat projectionMatrix = get_projection_matrix(pose);
    cv::Mat imagePoint = projectionMatrix * pointMat;
    imagePoint /= imagePoint.at<double>(2, 0);
    return cv::Point2f(imagePoint.at<double>(0, 0), imagePoint.at<double>(1, 0));
}

int Camera::get_width() const { return m_width; }

int Camera::get_height() const { return m_height; }

const cv::Mat &Camera::get_intrinsic_matrix() const { return m_K; }

bool Camera::is_visible(const cv::Point2f &point) const
{
    return point.x >= 0 && point.x < m_width && point.y >= 0 && point.y < m_height;
}

}; // namespace slam