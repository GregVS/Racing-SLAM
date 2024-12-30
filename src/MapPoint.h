#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

class Frame;

class MapPoint {
public:
    MapPoint(int id, cv::Point3f position);

    float orbDistance(cv::Mat descriptor) const;

    void addObservation(Frame *frame, int keypointIndex);

    int getId() const;

    const cv::Point3f &getPosition() const;

private:
    const int id;
    const cv::Point3f position;
    std::vector<Frame *> observationFrames;
    std::vector<int> observationKeypointIndices;
};
