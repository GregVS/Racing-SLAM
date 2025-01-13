#include "Map.h"

#include "Frame.h"

namespace slam {

// Map Point
MapPoint::MapPoint(const Eigen::Vector3f& position) : m_position(position)
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

// Map
Map::Map()
{
}

void Map::add_point(const Eigen::Vector3f& position)
{
    m_points.insert(std::make_unique<MapPoint>(position));
}

void Map::add_point(std::unique_ptr<MapPoint>&& point)
{
    m_points.insert(std::move(point));
}

void Map::remove_point(MapPoint* point)
{
    for (const auto& map_point : m_points) {
        if (map_point.get() == point) {
            for (const auto& [frame, index] : map_point->observations()) {
                const_cast<Frame*>(frame)->remove_map_match({*map_point, index});
            }
            m_points.erase(map_point);
            break;
        }
    }
}

// Const Map Point Iterator
Map::const_iterator Map::begin() const
{
    return const_iterator(m_points.begin());
}

Map::const_iterator Map::end() const
{
    return const_iterator(m_points.end());
}

Map::const_iterator::const_iterator(
    std::unordered_set<std::unique_ptr<MapPoint>>::const_iterator it)
    : m_it(it)
{
}

const MapPoint& Map::const_iterator::operator*() const
{
    return **m_it;
}

Map::const_iterator& Map::const_iterator::operator++()
{
    ++m_it;
    return *this;
}

bool Map::const_iterator::operator!=(const const_iterator& other) const
{
    return m_it != other.m_it;
}

// Mutable Map Point Iterator
Map::iterator Map::begin()
{
    return iterator(m_points.begin());
}

Map::iterator Map::end()
{
    return iterator(m_points.end());
}

Map::iterator::iterator(std::unordered_set<std::unique_ptr<MapPoint>>::iterator it) : m_it(it)
{
}

MapPoint& Map::iterator::operator*() const
{
    return **m_it;
}

Map::iterator& Map::iterator::operator++()
{
    ++m_it;
    return *this;
}

bool Map::iterator::operator!=(const iterator& other) const
{
    return m_it != other.m_it;
}

} // namespace slam