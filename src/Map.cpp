#include "Map.h"

Map::Map(const Camera &camera)
    : camera(camera)
{
}

Frame &Map::addFrame(Frame &&frame)
{
    frames.push_back(std::make_unique<Frame>(std::move(frame)));
    return *frames.back();
}

const std::vector<std::unique_ptr<Frame> > &Map::getFrames() const
{
    return frames;
}

MapPoint &Map::getMapPoint(int id)
{
    return points.at(id);
}

MapPoint &Map::addMapPoint(cv::Point3f position)
{
    int id = nextPointId;
    points.insert({ id, MapPoint(id, position) });
    nextPointId++;
    return points.at(id);
}

int Map::getNextFrameId() const
{
    return frames.size();
}

Frame &Map::getLastFrame()
{
    return *frames.back();
}

std::unordered_map<int, MapPoint> &Map::getMapPoints()
{
    return points;
}

const std::unordered_map<int, MapPoint> &Map::getMapPoints() const
{
    return points;
}

const Camera &Map::getCamera() const
{
    return camera;
}
