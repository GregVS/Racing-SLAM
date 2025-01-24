#include "Triangulation.h"

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "Frame.h"

namespace slam::triangulation {

std::pair<std::vector<Eigen::Vector2f>, std::vector<Eigen::Vector2f>>
get_matching_points(const ExtractedFeatures& features1,
                    const ExtractedFeatures& features2,
                    const std::vector<FeatureMatch>& matches)
{
    std::vector<Eigen::Vector2f> points1;
    std::vector<Eigen::Vector2f> points2;
    for (int i = 0; i < matches.size(); i++) {
        const auto& p1_cv = features1.keypoints[matches[i].train_index].pt;
        const auto& p2_cv = features2.keypoints[matches[i].query_index].pt;

        points1.push_back(Eigen::Vector2f(p1_cv.x, p1_cv.y));
        points2.push_back(Eigen::Vector2f(p2_cv.x, p2_cv.y));
    }
    return std::make_pair(points1, points2);
}

std::vector<TriangulatedPoint> triangulate_points(const Frame& frame1,
                                                  const Frame& frame2,
                                                  const std::vector<FeatureMatch>& matches,
                                                  const Camera& camera)
{
    auto [points1, points2] = get_matching_points(frame1.features(), frame2.features(), matches);
    return triangulate_points(points1, points2, frame1.pose(), frame2.pose(), camera);
}

std::vector<TriangulatedPoint> triangulate_points(const std::vector<Eigen::Vector2f>& points1,
                                                  const std::vector<Eigen::Vector2f>& points2,
                                                  const Eigen::Matrix4f& pose1,
                                                  const Eigen::Matrix4f& pose2,
                                                  const Camera& camera)
{
    // Convert the projection matrices to OpenCV format
    cv::Mat projection1_cv = cv_utils::projection_mat_cv(camera, pose1);
    cv::Mat projection2_cv = cv_utils::projection_mat_cv(camera, pose2);

    // Convert the points to OpenCV format
    std::vector<cv::Point2f> points1_cv;
    std::vector<cv::Point2f> points2_cv;
    for (int i = 0; i < points1.size(); i++) {
        points1_cv.push_back(cv::Point2f(points1[i][0], points1[i][1]));
        points2_cv.push_back(cv::Point2f(points2[i][0], points2[i][1]));
    }

    // Triangulate the points
    cv::Mat points4D;
    cv::triangulatePoints(projection1_cv, projection2_cv, points1_cv, points2_cv, points4D);

    // Convert the points to 3D and filter
    std::vector<TriangulatedPoint> triangulated;
    for (int i = 0; i < points4D.cols; i++) {
        auto point = Eigen::Vector3f(
            points4D.col(i).at<float>(0, 0) / points4D.col(i).at<float>(3, 0),
            points4D.col(i).at<float>(1, 0) / points4D.col(i).at<float>(3, 0),
            points4D.col(i).at<float>(2, 0) / points4D.col(i).at<float>(3, 0));

        Eigen::Vector3f cam1_point = pose1.block<3, 4>(0, 0) * point.homogeneous();
        Eigen::Vector3f cam2_point = pose2.block<3, 4>(0, 0) * point.homogeneous();

        // Filter points that are behind either camera
        if (cam1_point.z() < 0 || cam2_point.z() < 0) {
            continue;
        }

        // Filter points with very low parallax
        auto point_to_cam1 = pose1.inverse().block<3, 1>(0, 3) - point;
        auto point_to_cam2 = pose2.inverse().block<3, 1>(0, 3) - point;
        auto similarity = point_to_cam1.normalized().dot(point_to_cam2.normalized());
        if (similarity > 0.9999) {
            continue;
        }

        // Convert to image coordinates
        Eigen::Vector2f image1_point = (camera.get_intrinsic_matrix() * cam1_point).hnormalized();
        Eigen::Vector2f image2_point = (camera.get_intrinsic_matrix() * cam2_point).hnormalized();

        // Filter points with poor reprojection error
        auto reprojection_error1 = (image1_point - points1[i]).norm();
        auto reprojection_error2 = (image2_point - points2[i]).norm();
        if (reprojection_error1 > 2 || reprojection_error2 > 2) {
            continue;
        }

        triangulated.push_back(TriangulatedPoint{.position = point, .match_index = i});
    }

    return triangulated;
}

} // namespace slam::triangulation