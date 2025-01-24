#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace slam {

class Camera {
  public:
    Camera(float focal_len, int width, int height);
    Camera(float fx, float fy, float cx, float cy, int width, int height);

    const Eigen::Matrix3f& get_intrinsic_matrix() const;

    int get_width() const;
    int get_height() const;

    Eigen::Vector2f project(const Eigen::Matrix4f& pose, const Eigen::Vector3f& point) const;
    bool is_in_image(const Eigen::Vector2f& uv) const;

  private:
    const Eigen::Matrix3f m_K;
    const int m_width;
    const int m_height;
};

namespace cv_utils {

cv::Mat projection_mat_cv(const Camera& camera, const Eigen::Matrix4f& pose);

cv::Mat intrinsic_mat_cv(const Camera& camera);

cv::Mat extrinsic_mat_cv(const Eigen::Matrix4f& extrinsic);

} // namespace camera

}; // namespace slam