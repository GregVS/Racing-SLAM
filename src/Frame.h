#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "MapPoint.h"

class Frame {
public:
    Frame(int id, const cv::Mat &image, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors);

    bool hasCorrespondingMapPoint(int keypointIndex) const;

    MapPoint* getCorrespondingMapPoint(int keypointIndex) const;

    void setCorrespondingMapPoint(int keypointIndex, MapPoint* mapPoint);

    cv::Mat getDescriptor(int keypointIndex) const;

    const cv::Mat &getDescriptors() const;

    const cv::Mat &getImage() const;

    const std::vector<cv::KeyPoint> &getKeypoints() const;

    const cv::KeyPoint &getKeypoint(int keypointIndex) const;

    const cv::Mat &getPose() const;

    void setPose(const cv::Mat &pose);

    int getId() const;

private:
    const int id;
    const cv::Mat image;

    const std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    cv::Mat pose;
    std::vector<MapPoint*> mapPoints;
};
