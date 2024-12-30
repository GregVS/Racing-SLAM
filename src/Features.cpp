#include "Features.h"
#include <unordered_set>

Frame extractFeatures(const cv::Mat &image, int id)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gray, corners, 3000, 0.01, 8);

    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints;
    for (const auto &corner : corners) {
        keypoints.push_back(cv::KeyPoint(corner, 20.0f));
    }

    cv::Mat descriptors;
    orb->compute(image, keypoints, descriptors);

    return Frame(id, image, keypoints, descriptors);
}

std::vector<cv::DMatch> matchFeatures(const Frame &prevFrame, const Frame &frame)
{
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch(frame.getDescriptors(), prevFrame.getDescriptors(), knn_matches, 2);

    std::unordered_set<int> matchedKeypoints;

    std::vector<cv::DMatch> good_matches;
    for (const auto &match : knn_matches) {
        // Ratio test
        if (matchedKeypoints.find(match[0].queryIdx) != matchedKeypoints.end() ||
            matchedKeypoints.find(match[0].trainIdx) != matchedKeypoints.end()) {
            continue;
        }

        if (match[0].distance < 0.75 * match[1].distance && match[0].distance < 32) {
            good_matches.push_back(match[0]);
            matchedKeypoints.insert(match[0].queryIdx);
            matchedKeypoints.insert(match[0].trainIdx);
        }
    }

    return good_matches;
}

PoseEstimate estimatePose(const Frame &prevFrame, const Frame &frame, const Camera &camera,
                          const std::vector<cv::DMatch> &matches)
{
    // Keypoints based on matches
    std::vector<cv::Point2f> fromPoints, toPoints;
    for (const auto &match : matches) {
        fromPoints.push_back(prevFrame.getKeypoint(match.trainIdx).pt);
        toPoints.push_back(frame.getKeypoint(match.queryIdx).pt);
    }

    // Essential matrix and pose estimation
    std::vector<uchar> inliers;
    cv::Mat E =
            cv::findEssentialMat(fromPoints, toPoints, camera.getIntrinsicMatrix(), cv::RANSAC, 0.999, 1.0, inliers);
    cv::Mat R, t;

    // Filter matches based on inliers
    std::vector<cv::DMatch> filteredMatches;
    for (int i = 0; i < inliers.size(); i++) {
        if (inliers[i] == 0) {
            continue;
        }
        filteredMatches.push_back(matches[i]);
    }
    std::cout << "Number of matches: " << filteredMatches.size() << std::endl;

    cv::recoverPose(E, fromPoints, toPoints, camera.getIntrinsicMatrix(), R, t, inliers);

    cv::Mat relativePose = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(relativePose(cv::Rect(0, 0, 3, 3)));
    t.copyTo(relativePose(cv::Rect(3, 0, 1, 3)));

    return PoseEstimate{ prevFrame.getPose() * relativePose.inv(), filteredMatches };
}
