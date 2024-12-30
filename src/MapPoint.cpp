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
    for (int i = 0; i < observationFrames.size(); i++) {
        int keypointIndex = observationKeypointIndices[i];
        float dist = cv::norm(observationFrames[i]->getDescriptor(keypointIndex), descriptor, cv::NORM_HAMMING);
        if (dist < minDist) {
            minDist = dist;
        }
    }
    return minDist;
}

void MapPoint::addObservation(Frame *frame, int keypointIndex)
{
    observationFrames.push_back(frame);
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
