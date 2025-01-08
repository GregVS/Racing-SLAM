#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "Triangulation.h"

namespace slam {

std::vector<Eigen::Vector3f> triangulate_features(
    const ExtractedFeatures& features1,
    const ExtractedFeatures& features2,
    const std::vector<FeatureMatch>& matches,
    const Eigen::Matrix4f& pose1,
    const Eigen::Matrix4f& pose2,
    const Camera& camera)
{
    // Collect the keypoints from the matches
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    for (const auto& match : matches) {
        points1.push_back(features1.keypoints[match.train_index].pt);
        points2.push_back(features2.keypoints[match.query_index].pt);
    }

    // Convert the projection matrices to OpenCV format
    cv::Mat projection1_cv = projection_mat_cv(camera, pose1);
    cv::Mat projection2_cv = projection_mat_cv(camera, pose2);

    // Triangulate the points
    cv::Mat points4D;
    cv::triangulatePoints(projection1_cv, projection2_cv, points1, points2, points4D);

    // Convert the points to 3D and filter
    std::vector<Eigen::Vector3f> points3D;
    for (int i = 0; i < points4D.cols; i++) {
        cv::Mat point = points4D.col(i);
        point /= point.at<float>(3, 0);

        cv::Mat cam1_points = projection1_cv * point;
        cv::Mat cam2_points = projection2_cv * point;

        // Filter points that are behind either camera
        if (cam1_points.at<float>(2, 0) < 0 || cam2_points.at<float>(2, 0) < 0) {
            continue;
        }
        std::cout << "point: " << point.at<float>(0, 0) << ", " << point.at<float>(1, 0) << ", " << point.at<float>(2, 0) << std::endl;

        points3D.push_back(Eigen::Vector3f(point.at<float>(0, 0), point.at<float>(1, 0), point.at<float>(2, 0)));
    }

    return points3D;
}

} // namespace slam