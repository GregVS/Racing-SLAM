#include "map.h"

Frame::Frame(int id, const cv::Mat &image, const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors)
        : id(id)
        , image(image)
        , keypoints(keypoints)
        , descriptors(descriptors)
{
    pose = cv::Mat::eye(4, 4, CV_64F);
    mapPointIndices.resize(keypoints.size(), -1);
}

bool Frame::hasCorrespondingMapPoint(int keypointIndex) const
{
    return mapPointIndices[keypointIndex] != -1;
}

int Frame::getCorrespondingMapPoint(int keypointIndex) const
{
    return mapPointIndices[keypointIndex];
}

void Frame::setCorrespondingMapPoint(int keypointIndex, int mapPointId)
{
    mapPointIndices[keypointIndex] = mapPointId;
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

MapPoint::MapPoint(int id, cv::Point3f position)
        : id(id)
        , position(position)
{
}

float MapPoint::orbDistance(cv::Mat descriptor, const Map &map) const
{
    float minDist = std::numeric_limits<float>::max();
    for (int i = 0; i < observationFrameIndices.size(); i++) {
        int frameIndex = observationFrameIndices[i];
        int keypointIndex = observationKeypointIndices[i];
        float dist = cv::norm(map.getFrames()[frameIndex]->getDescriptor(keypointIndex), descriptor, cv::NORM_HAMMING);
        if (dist < minDist) {
            minDist = dist;
        }
    }
    return minDist;
}

void MapPoint::addObservation(int frameIndex, int keypointIndex)
{
    observationFrameIndices.push_back(frameIndex);
    observationKeypointIndices.push_back(keypointIndex);
}

int MapPoint::getId() const
{
    return id;
}

const cv::Point3f &MapPoint::getPosition() const
{
    return position;
}

Map::Map()
{
}

Frame &Map::addFrame(const Frame &frame)
{
    frames.push_back(std::make_unique<Frame>(frame));
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

const MapPoint &Map::addMapPoint(cv::Point3f position)
{
    int id = nextPointId;
    points.insert({ id, MapPoint(id, position) });
    nextPointId++;
    return points.at(id);
}

void Map::addObservation(int frameIndex, int keypointIndex, int mapPointId)
{
    points.at(mapPointId).addObservation(frameIndex, keypointIndex);
    frames[frameIndex]->setCorrespondingMapPoint(keypointIndex, mapPointId);
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
