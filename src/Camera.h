#pragma once

#include <opencv2/opencv.hpp>

class Camera {
public:
    Camera(cv::Mat K, int width, int height)
            : K(K)
            , width(width)
            , height(height)
    {
    }

    cv::Mat getProjectionMatrix(const cv::Mat &pose) const
    {
        cv::Mat E = pose.inv();
        cv::Mat projectionMatrix = K * E.rowRange(0, 3);
        return projectionMatrix;
    }

    cv::Point3f toCameraCoordinates(const cv::Point3f &point, const cv::Mat &pose) const
    {
        cv::Mat pointMat = cv::Mat(4, 1, CV_64F);
        pointMat.at<double>(0, 0) = point.x;
        pointMat.at<double>(1, 0) = point.y;
        pointMat.at<double>(2, 0) = point.z;
        pointMat.at<double>(3, 0) = 1;

        cv::Mat cameraPoint = pose * pointMat;
        cameraPoint /= cameraPoint.at<double>(3, 0);
        return cv::Point3f(cameraPoint.at<double>(0, 0), cameraPoint.at<double>(1, 0), cameraPoint.at<double>(2, 0));
    }

    cv::Point2f toImageCoordinates(const cv::Point3f &point, const cv::Mat &pose) const
    {
        cv::Mat pointMat = cv::Mat(4, 1, CV_64F);
        pointMat.at<double>(0, 0) = point.x;
        pointMat.at<double>(1, 0) = point.y;
        pointMat.at<double>(2, 0) = point.z;
        pointMat.at<double>(3, 0) = 1;

        cv::Mat projectionMatrix = getProjectionMatrix(pose);
        cv::Mat imagePoint = projectionMatrix * pointMat;
        imagePoint /= imagePoint.at<double>(2, 0);
        return cv::Point2f(imagePoint.at<double>(0, 0), imagePoint.at<double>(1, 0));
    }

    int getWidth() const
    {
        return width;
    }

    int getHeight() const
    {
        return height;
    }

    const cv::Mat &getIntrinsicMatrix() const
    {
        return K;
    }

private:
    const cv::Mat K;
    const int width;
    const int height;
};
