#pragma once

#include <opencv2/opencv.hpp>
#include <unordered_map>

class Frame;

class MapPoint {
public:
    MapPoint(int id, cv::Point3f position);

    float orbDistance(cv::Mat descriptor) const;

    void addObservation(Frame *frame, int keypointIndex);

    int getId() const;

    const cv::Point3f &getPosition() const;

    const std::unordered_map<Frame *, int> &getObservations() const;

    bool isObservedBy(Frame *frame) const;

private:
    const int id;
    const cv::Point3f position;
    std::unordered_map<Frame *, int> observations;
};
