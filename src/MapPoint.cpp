#include "MapPoint.h"
#include "Frame.h"

namespace slam
{

MapPoint::MapPoint(int id, cv::Point3f position)
        : m_id(id)
        , m_position(position)
{
}

void MapPoint::add_observation(Frame *frame, int keypointIndex)
{
    m_observations.insert(std::make_pair(frame, keypointIndex));
}

int MapPoint::get_id() const
{
    return m_id;
}

const cv::Point3f &MapPoint::get_position() const
{
    return m_position;
}

std::vector<MapPoint::ObservationData> MapPoint::get_observations_vector() const
{
    return std::vector<ObservationData>(m_observations.begin(), m_observations.end());
}

void MapPoint::for_each_observation(const std::function<void(Frame *, int)> &callback) const
{
    for (const auto &[frame, keypointIndex] : m_observations) {
        callback(frame, keypointIndex);
    }
}

bool MapPoint::is_observed_by(Frame *frame) const
{
    return m_observations.find(frame) != m_observations.end();
}

float map_point_orb_distance(const MapPoint &map_point, const cv::Mat &descriptor)
{
    float minDist = std::numeric_limits<float>::max();
    map_point.for_each_observation([&](Frame *frame, int keypointIndex) {
        float dist = cv::norm(frame->get_descriptor(keypointIndex), descriptor, cv::NORM_HAMMING);
        if (dist < minDist) {
            minDist = dist;
        }
    });
    return minDist;
}

};