#include "MapPoint.h"
#include "Frame.h"

MapPoint::MapPoint(int id, cv::Point3f position)
        : id(id)
        , position(position)
{
}

float MapPoint::orbDistance(cv::Mat descriptor) const
{
    float minDist = std::numeric_limits<float>::max();
    for (const auto &[frame, keypointIndex] : observations) {
        float dist = cv::norm(frame->getDescriptor(keypointIndex), descriptor, cv::NORM_HAMMING);
        if (dist < minDist) {
            minDist = dist;
        }
    }
    return minDist;
}

void MapPoint::addObservation(Frame *frame, int keypointIndex)
{
    observations.insert(std::make_pair(frame, keypointIndex));
}

int MapPoint::getId() const
{
    return id;
}

const cv::Point3f &MapPoint::getPosition() const
{
    return position;
}

const std::unordered_map<Frame *, int> &MapPoint::getObservations() const
{
    return observations;
}

bool MapPoint::isObservedBy(Frame *frame) const
{
    return observations.find(frame) != observations.end();
}
