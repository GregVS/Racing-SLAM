#include "MapPoint.h"
#include "Frame.h"

using slam::Frame;
using slam::MapPoint;

MapPoint::MapPoint(const Eigen::Vector3f& position) : m_position(position), m_color(255, 255, 255)
{

}

const Eigen::Vector3f& MapPoint::position() const
{
    return m_position;
}

void MapPoint::set_position(const Eigen::Vector3f& position)
{
    m_position = position;
}

void MapPoint::add_observation(const Frame* key_frame, size_t index)
{
    m_observations[key_frame] = index;
}

const std::unordered_map<const Frame*, size_t>& MapPoint::observations() const
{
    return m_observations;
}

const cv::Vec3b& MapPoint::color() const
{
    return m_color;
}

void MapPoint::set_color(const cv::Vec3b& color)
{
    m_color = color;
}

