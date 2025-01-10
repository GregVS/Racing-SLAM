#include "Triangulation.h"

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

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

static float reprojection_error(const cv::Point2f& point, const cv::Mat& projection)
{
    return (point.x - projection.at<float>(0, 0)) * (point.x - projection.at<float>(0, 0)) +
           (point.y - projection.at<float>(1, 0)) * (point.y - projection.at<float>(1, 0));
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
        cv::Mat point = points4D.col(i);
        point /= point.at<float>(3, 0);

        cv::Mat cam1_point = cv_utils::extrinsic_mat_cv(pose1) * point;
        cv::Mat cam2_point = cv_utils::extrinsic_mat_cv(pose2) * point;

        // Filter points that are behind either camera
        if (cam1_point.at<float>(2, 0) < 0 || cam2_point.at<float>(2, 0) < 0) {
            continue;
        }

        // Convert to image coordinates
        cv::Mat image1_point = cv_utils::intrinsic_mat_cv(camera) * cam1_point;
        cv::Mat image2_point = cv_utils::intrinsic_mat_cv(camera) * cam2_point;

        image1_point /= image1_point.at<float>(2, 0);
        image2_point /= image2_point.at<float>(2, 0);

        // Filter points with poor reprojection error
        auto reprojection_error1 = reprojection_error(points1_cv[i], image1_point);
        auto reprojection_error2 = reprojection_error(points2_cv[i], image2_point);
        if (reprojection_error1 > 2 || reprojection_error2 > 2) {
            continue;
        }

        triangulated.push_back(TriangulatedPoint{
            .position = Eigen::Vector3f(point.at<float>(0, 0), point.at<float>(1, 0), point.at<float>(2, 0)),
            .match_index = i
        });
    }

    return triangulated;
}

} // namespace slam