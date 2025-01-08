#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace slam {

class Camera {
  public:
    Camera(float focal_len, int width, int height);

    const Eigen::Matrix3f &get_intrinsic_matrix() const;

    int get_width() const;
    int get_height() const;

  private:
    const Eigen::Matrix3f m_K;
    const int m_width;
    const int m_height;
};


cv::Mat projection_mat_cv(const Camera &camera, const Eigen::Matrix4f &pose);

cv::Mat intrinsic_mat_cv(const Camera &camera);

}; // namespace slam