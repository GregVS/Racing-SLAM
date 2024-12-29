#pragma once

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <limits>

struct MapPoint;
struct Map;

class Frame {
    friend class Map;

public:
    Frame(int id, const cv::Mat &image, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors);

    bool hasCorrespondingMapPoint(int keypointIndex) const;

    int getCorrespondingMapPoint(int keypointIndex) const;

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
    cv::Mat pose;
    std::vector<int> mapPointIndices;
    cv::Mat descriptors;

    void setCorrespondingMapPoint(int keypointIndex, int mapPointId);
};

struct MapPoint {
    friend class Map;

public:
    MapPoint(int id, cv::Point3f position);

    float orbDistance(cv::Mat descriptor, const Map &map) const;

    int getId() const;

    const cv::Point3f &getPosition() const;

private:
    const int id;
    const cv::Point3f position;
    std::vector<int> observationFrameIndices;
    std::vector<int> observationKeypointIndices;

    void addObservation(int frameIndex, int keypointIndex);
};

class Map {
public:
    Map();

    Frame &addFrame(const Frame &frame);

    Frame &getLastFrame();

    MapPoint &getMapPoint(int id);

    const MapPoint &addMapPoint(cv::Point3f position);

    const std::vector<std::unique_ptr<Frame> > &getFrames() const;

    int getNextFrameId() const;

    std::unordered_map<int, MapPoint> &getMapPoints();

    const std::unordered_map<int, MapPoint> &getMapPoints() const;

    void addObservation(int frameIndex, int keypointIndex, int mapPointId);

private:
    std::unordered_map<int, MapPoint> points;
    std::vector<std::unique_ptr<Frame> > frames;

    int nextPointId = 0;
};

struct Camera {
    cv::Mat K;
    int width, height;
};
