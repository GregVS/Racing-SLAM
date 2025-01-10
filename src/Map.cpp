#include "Map.h"

namespace slam {

MapPoint::MapPoint(const Eigen::Vector3f& position) : m_position(position) {}

const Eigen::Vector3f& MapPoint::position() const { return m_position; }

void MapPoint::add_observation(const std::shared_ptr<KeyFrame>& key_frame, int index)
{
    m_observations[key_frame] = index;
}

const std::unordered_map<std::shared_ptr<KeyFrame>, int>& MapPoint::observations() const
{
    return m_observations;
}

Map::Map() {}

void Map::add_point(const Eigen::Vector3f& position)
{
    m_points.insert(std::make_unique<MapPoint>(position));
}

void Map::add_point(std::unique_ptr<MapPoint>&& point) { m_points.insert(std::move(point)); }

Map::iterator Map::begin() const { return iterator(m_points.begin()); }

Map::iterator Map::end() const { return iterator(m_points.end()); }

Map::iterator::iterator(std::unordered_set<std::unique_ptr<MapPoint>>::const_iterator it) : m_it(it) {}

MapPoint& Map::iterator::operator*() const { return **m_it; }

Map::iterator& Map::iterator::operator++()
{
    ++m_it;
    return *this;
}

bool Map::iterator::operator!=(const iterator& other) const { return m_it != other.m_it; }

} // namespace slam