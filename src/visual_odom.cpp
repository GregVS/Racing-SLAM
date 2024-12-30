#include "visual_odom.h"
#include <vector>

Frame extractFeatures(const cv::Mat &image, int id)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    return Frame(id, image, keypoints, descriptors);
}

std::vector<cv::DMatch> matchFeatures(const Frame &prevFrame, const Frame &frame)
{
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch(frame.getDescriptors(), prevFrame.getDescriptors(), knn_matches, 2);

    std::vector<cv::DMatch> good_matches;
    for (const auto &match : knn_matches) {
        // Ratio test
        if (match[0].distance < 0.75 * match[1].distance && match[0].distance < 32) {
            good_matches.push_back(match[0]);
        }
    }

    std::cout << "Number of matches: " << good_matches.size() << std::endl;

    return good_matches;
}

cv::Mat estimatePose(const Frame &prevFrame, const Frame &frame, const std::vector<cv::DMatch> &matches, const cv::Mat &K)
{
    // Keypoints based on matches
    std::vector<cv::Point2f> fromPoints, toPoints;
    for (const auto &match : matches) {
        fromPoints.push_back(prevFrame.getKeypoint(match.trainIdx).pt);
        toPoints.push_back(frame.getKeypoint(match.queryIdx).pt);
    }

    // Essential matrix and pose estimation
    cv::Mat E = cv::findEssentialMat(fromPoints, toPoints, K, cv::RANSAC);
    cv::Mat R, t;
    cv::recoverPose(E, fromPoints, toPoints, K, R, t);

    cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(pose(cv::Rect(0, 0, 3, 3)));
    t.copyTo(pose(cv::Rect(3, 0, 1, 3)));

    cv::Mat camera_pose = prevFrame.getPose() * pose.inv();

    return camera_pose;
}

void matchMapPoints(Map &map, Frame &frame)
{
    int matches = 0;
    for (auto &[_, point] : map.getMapPoints()) {
        // Project point into frame
        cv::Mat projectionMatrix = map.getCamera().getProjectionMatrix(frame.getPose());
        cv::Mat point4D = cv::Mat(4, 1, CV_64F);
        point4D.at<double>(0, 0) = point.getPosition().x;
        point4D.at<double>(1, 0) = point.getPosition().y;
        point4D.at<double>(2, 0) = point.getPosition().z;
        point4D.at<double>(3, 0) = 1;
        cv::Mat pointInFrame = projectionMatrix * point4D;
        pointInFrame /= pointInFrame.at<double>(2, 0);

        // Check if point is in frame
        auto point2D = cv::Point2f(pointInFrame.at<double>(0, 0), pointInFrame.at<double>(1, 0));
        if (point2D.x < 0 || point2D.x > map.getCamera().getWidth() || point2D.y < 0 || point2D.y > map.getCamera().getHeight()) {
            continue;
        }

        // Find a match
        for (int i = 0; i < frame.getKeypoints().size(); i++) {
            if (cv::norm(frame.getKeypoint(i).pt - point2D) < 10) {
                float orbDist = point.orbDistance(frame.getDescriptor(i));
                if (orbDist < 32) {
                    point.addObservation(&frame, i);
                    frame.setCorrespondingMapPoint(i, &point);
                    matches++;
                    break;
                }
            }
        }
    }

    std::cout << "Matches: " << matches << std::endl;
}

void triangulatePoints(Map &map, Frame &prevFrame, Frame &frame, const std::vector<cv::DMatch> &matches)
{
    std::vector<cv::Point2f> fromPoints, toPoints;
    std::vector<int> frameKeypointIndices;
    std::vector<int> prevFrameKeypointIndices;
    for (const auto &match : matches) {
        fromPoints.push_back(prevFrame.getKeypoint(match.trainIdx).pt);
        toPoints.push_back(frame.getKeypoint(match.queryIdx).pt);
        frameKeypointIndices.push_back(match.queryIdx);
        prevFrameKeypointIndices.push_back(match.trainIdx);
    }

    // Compute the projection matrices
    cv::Mat P1 = map.getCamera().getProjectionMatrix(prevFrame.getPose());
    cv::Mat P2 = map.getCamera().getProjectionMatrix(frame.getPose());

    // Triangulate
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, fromPoints, toPoints, points4D);

    // Convert the points to 3D and compute reprojection errors
    int addedPoints = 0;
    std::vector<float> reprojection_errors;

    for (int i = 0; i < points4D.cols; i++) {
        cv::Point3f point3D;
        point3D.x = points4D.at<float>(0, i) / points4D.at<float>(3, i);
        point3D.y = points4D.at<float>(1, i) / points4D.at<float>(3, i);
        point3D.z = points4D.at<float>(2, i) / points4D.at<float>(3, i);

        if (frame.hasCorrespondingMapPoint(frameKeypointIndices[i])) {
            // Point already matched to a 3D map point
            continue;
        } else if (prevFrame.hasCorrespondingMapPoint(prevFrameKeypointIndices[i])) {
            // Previous frame keypoints already matched to a 3D map point
            MapPoint *mapPoint = prevFrame.getCorrespondingMapPoint(prevFrameKeypointIndices[i]);
            mapPoint->addObservation(&frame, frameKeypointIndices[i]);
            frame.setCorrespondingMapPoint(frameKeypointIndices[i], mapPoint);
            continue;
        }

        // Check if point is in front of camera
        auto point_in_camera_1 = map.getCamera().toCameraCoordinates(point3D, prevFrame.getPose());
        auto point_in_camera_2 = map.getCamera().toCameraCoordinates(point3D, frame.getPose());
        if (point_in_camera_1.z < 0 || point_in_camera_2.z < 0) {
            continue;
        }

        // Compute reprojection errors
        auto reprojection1 = map.getCamera().toImageCoordinates(point3D, prevFrame.getPose());
        auto reprojection2 = map.getCamera().toImageCoordinates(point3D, frame.getPose());

        reprojection_errors.push_back(cv::norm(reprojection1 - prevFrame.getKeypoint(matches[i].trainIdx).pt));
        reprojection_errors.push_back(cv::norm(reprojection2 - frame.getKeypoint(matches[i].queryIdx).pt));

        auto &mapPoint = map.addMapPoint(point3D);
        mapPoint.addObservation(&frame, frameKeypointIndices[i]);
        mapPoint.addObservation(&prevFrame, prevFrameKeypointIndices[i]);
        frame.setCorrespondingMapPoint(frameKeypointIndices[i], &mapPoint);
        prevFrame.setCorrespondingMapPoint(prevFrameKeypointIndices[i], &mapPoint);
        addedPoints++;
    }

    std::cout << "Reprojection error: " << cv::mean(reprojection_errors)[0] << std::endl;
    std::cout << "New points: " << addedPoints << std::endl;
}
