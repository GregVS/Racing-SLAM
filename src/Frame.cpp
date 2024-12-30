#include "Frame.h"

Frame::Frame(int id, const cv::Mat &image, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors)
        : id(id)
        , image(image)
        , keypoints(keypoints)
        , descriptors(descriptors)
{
    pose = cv::Mat::eye(4, 4, CV_64F);
    mapPoints.resize(keypoints.size(), nullptr);
}

bool Frame::hasCorrespondingMapPoint(int keypointIndex) const
{
    return mapPoints[keypointIndex] != nullptr;
}

MapPoint *Frame::getCorrespondingMapPoint(int keypointIndex) const
{
    return mapPoints[keypointIndex];
}

void Frame::setCorrespondingMapPoint(int keypointIndex, MapPoint *mapPoint)
{
    mapPoints[keypointIndex] = mapPoint;
}

cv::Mat Frame::getDescriptor(int keypointIndex) const
{
    return descriptors.row(keypointIndex);
}

const cv::Mat &Frame::getDescriptors() const
{
    return descriptors;
}

const cv::Mat &Frame::getImage() const
{
    return image;
}

const std::vector<cv::KeyPoint> &Frame::getKeypoints() const
{
    return keypoints;
}

const cv::KeyPoint &Frame::getKeypoint(int keypointIndex) const
{
    return keypoints[keypointIndex];
}

const cv::Mat &Frame::getPose() const
{
    return pose;
}

void Frame::setPose(const cv::Mat &pose)
{
    this->pose = pose;
}

int Frame::getId() const
{
    return id;
}
