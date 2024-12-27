#include "visual_odom.h"
#include <vector>

FrameData extractFeatures(const cv::Mat &frame)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create(5000);
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

    cv::Mat camera_pose = prevFrame.pose * pose;

    return camera_pose;
}

