#include "visual_odom.h"
#include <vector>

FrameData extractFeatures(const cv::Mat &frame)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(frame, cv::noArray(), keypoints, descriptors);

    return {frame, keypoints, descriptors};
}

std::vector<cv::DMatch> matchFeatures(const FrameData &prevFrame, const FrameData &frame)
{
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(frame.descriptors, prevFrame.descriptors, knn_matches, 2);

    std::vector<cv::DMatch> good_matches;
    for(const auto& match : knn_matches) {
        // Ratio test
        if(match[0].distance < 0.75 * match[1].distance && match[0].distance < 32) {
            good_matches.push_back(match[0]);
        }
    }

    std::cout << "Number of matches: " << good_matches.size() << std::endl;

    return good_matches;
}

std::vector<cv::Point3f> triangulatePoints(const FrameData &prevFrame, const FrameData &frame, const std::vector<cv::DMatch> &matches)
{
    std::vector<cv::Point2f> fromPoints, toPoints;
    for (const auto &match : matches) {
        fromPoints.push_back(prevFrame.keypoints[match.trainIdx].pt);
        toPoints.push_back(frame.keypoints[match.queryIdx].pt);
    }

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 800;
    K.at<double>(1, 1) = 800;
    K.at<double>(0, 2) = 1920/2;
    K.at<double>(1, 2) = 1080/2;

    // Compute the projection matrices
    cv::Mat E1 = prevFrame.pose.inv();
    cv::Mat E2 = frame.pose.inv();
    cv::Mat P1 = K * E1.rowRange(0, 3);
    cv::Mat P2 = K * E2.rowRange(0, 3);

    // Triangulate
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, fromPoints, toPoints, points4D);

    // Convert the points to 3D and compute reprojection errors
    std::vector<cv::Point3f> points3D;
    std::vector<float> reprojection_errors;

    for (int i = 0; i < points4D.cols; i++) {
        cv::Point3f point3D;
        point3D.x = points4D.at<float>(0, i) / points4D.at<float>(3, i);
        point3D.y = points4D.at<float>(1, i) / points4D.at<float>(3, i);
        point3D.z = points4D.at<float>(2, i) / points4D.at<float>(3, i);

        // Check if point is in front of camera
        cv::Mat point = cv::Mat(4, 1, CV_64F);
        point.at<double>(0, 0) = point3D.x;
        point.at<double>(1, 0) = point3D.y;
        point.at<double>(2, 0) = point3D.z;
        point.at<double>(3, 0) = 1;

        cv::Mat point_in_camera_1 = prevFrame.pose.inv() * point;
        cv::Mat point_in_camera_2 = frame.pose.inv() * point;

        if (point_in_camera_1.at<double>(2, 0) < 0 || point_in_camera_2.at<double>(2, 0) < 0) {
            continue;
        }

        cv::Mat reprojection1 = P1 * point;
        cv::Mat reprojection2 = P2 * point;

        reprojection1 /= reprojection1.at<double>(2, 0);
        reprojection2 /= reprojection2.at<double>(2, 0);

        cv::Point2f reprojection1_2d = cv::Point2f(reprojection1.at<double>(0, 0), reprojection1.at<double>(1, 0));
        cv::Point2f reprojection2_2d = cv::Point2f(reprojection2.at<double>(0, 0), reprojection2.at<double>(1, 0));

        reprojection_errors.push_back(cv::norm(reprojection1_2d - prevFrame.keypoints[matches[i].trainIdx].pt));
        reprojection_errors.push_back(cv::norm(reprojection2_2d - frame.keypoints[matches[i].queryIdx].pt));

        points3D.push_back(point3D);
    }

    std::cout << "Reprojection error: " << cv::mean(reprojection_errors)[0] << std::endl;

    return points3D;
}

cv::Mat estimatePose(const FrameData &prevFrame, const FrameData &frame, const std::vector<cv::DMatch> &matches)
{
    // Keypoints based on matches
    std::vector<cv::Point2f> fromPoints, toPoints;
    for (const auto &match : matches) {
        fromPoints.push_back(prevFrame.keypoints[match.trainIdx].pt);
        toPoints.push_back(frame.keypoints[match.queryIdx].pt);
    }

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 800;
    K.at<double>(1, 1) = 800;
    K.at<double>(0, 2) = 1920/2;
    K.at<double>(1, 2) = 1080/2;

    // Essential matrix and pose estimation
    cv::Mat E = cv::findEssentialMat(fromPoints, toPoints, K, cv::RANSAC);
    cv::Mat R, t;
    cv::recoverPose(E, fromPoints, toPoints, K, R, t);

    cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(pose(cv::Rect(0, 0, 3, 3)));
    t.copyTo(pose(cv::Rect(3, 0, 1, 3)));

    cv::Mat camera_pose = prevFrame.pose * pose.inv();

    return camera_pose;
}

