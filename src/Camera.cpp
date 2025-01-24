#include "Camera.h"

namespace slam {

static Eigen::Matrix3f make_intrinsic_matrix(float fx, float fy, float cx, float cy)
{
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K(0, 0) = fx;
    K(1, 1) = fy;
    K(0, 2) = cx;
    K(1, 2) = cy;
    return K;
}

Camera::Camera(float focal_len, int width, int height)
    : m_K(make_intrinsic_matrix(focal_len, focal_len, width / 2, height / 2)), m_width(width), m_height(height)
{
}

Camera::Camera(float fx, float fy, float cx, float cy, int width, int height)
    : m_K(make_intrinsic_matrix(fx, fy, cx, cy)), m_width(width), m_height(height)
{
}

Eigen::Vector2f Camera::project(const Eigen::Matrix4f& pose, const Eigen::Vector3f& point) const
{
    Eigen::Vector3f uv = m_K * pose.block<3, 4>(0, 0) * point.homogeneous();
    return uv.hnormalized();
}

bool Camera::is_in_image(const Eigen::Vector2f& uv) const
{
    return uv[0] >= 0 && uv[0] < m_width && uv[1] >= 0 && uv[1] < m_height;
}

int Camera::get_width() const { return m_width; }

int Camera::get_height() const { return m_height; }

const Eigen::Matrix3f& Camera::get_intrinsic_matrix() const { return m_K; }

namespace cv_utils {

cv::Mat projection_mat_cv(const Camera& camera, const Eigen::Matrix4f& pose)
{
    Eigen::Matrix<float, 3, 4> projection = camera.get_intrinsic_matrix() * pose.block<3, 4>(0, 0);
    cv::Mat projection_cv = cv::Mat(3, 4, CV_32F);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            projection_cv.at<float>(i, j) = projection(i, j);
        }
    }
    return projection_cv;
}

cv::Mat intrinsic_mat_cv(const Camera& camera)
{
    Eigen::Matrix<float, 3, 3> intrinsic = camera.get_intrinsic_matrix();
    cv::Mat intrinsic_cv = cv::Mat(3, 3, CV_32F);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            intrinsic_cv.at<float>(i, j) = intrinsic(i, j);
        }
    }
    return intrinsic_cv;
}

cv::Mat extrinsic_mat_cv(const Eigen::Matrix4f& extrinsic)
{
    cv::Mat extrinsic_cv = cv::Mat(3, 4, CV_32F);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            extrinsic_cv.at<float>(i, j) = extrinsic(i, j);
        }
    }
    return extrinsic_cv;
}

} // namespace camera

} // namespace slam