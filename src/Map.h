#pragma once

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <limits>
#include "Frame.h"
#include "Camera.h"

struct MapPoint;
struct Map;

class Map {
public:
    Map(const Camera &camera);

    Frame &addFrame(Frame &&frame);

    Frame &getLastFrame();

    MapPoint &getMapPoint(int id);

    MapPoint &addMapPoint(cv::Point3f position);

    const std::vector<std::unique_ptr<Frame> > &getFrames() const;

    int getNextFrameId() const;

    std::unordered_map<int, MapPoint> &getMapPoints();

    const std::unordered_map<int, MapPoint> &getMapPoints() const;

    const Camera &getCamera() const;

private:
    std::unordered_map<int, MapPoint> points;
    std::vector<std::unique_ptr<Frame> > frames;
    Camera camera;

    int nextPointId = 0;
};