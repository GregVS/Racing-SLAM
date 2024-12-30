#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "MapPoint.h"
#include "KDTree.h"
class Frame {
public:
    Frame(int id, const cv::Mat &image, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors);

    Frame(const Frame &other) = delete;

    Frame(Frame &&other) noexcept = default;

    bool hasCorrespondingMapPoint(int keypointIndex) const;

    MapPoint *getCorrespondingMapPoint(int keypointIndex) const;

    void setCorrespondingMapPoint(int keypointIndex, MapPoint *mapPoint);

    cv::Mat getDescriptor(int keypointIndex) const;

    const cv::Mat &getDescriptors() const;

    const cv::Mat &getImage() const;

    const std::vector<cv::KeyPoint> &getKeypoints() const;

    const cv::KeyPoint &getKeypoint(int keypointIndex) const;

    const cv::Mat &getPose() const;

    std::vector<size_t> getKeypointsWithinRadius(const cv::Point2f &target, float radius) const;

    void setPose(const cv::Mat &pose);

    int getId() const;

private:
    const int id;
    const cv::Mat image;

    const std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    KDTree2D kdTree;

    cv::Mat pose;
    std::vector<MapPoint *> mapPoints;
};
