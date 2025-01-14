#include "Map.h"
#include "Frame.h"

namespace slam {

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

void Map::create_point(const Eigen::Vector3f& position, Frame& frame1, Frame& frame2, FeatureMatch& match)
{
    auto point = std::make_unique<MapPoint>(position);

    // Add associations
    point->add_observation(&frame1, match.train_index);
    point->add_observation(&frame2, match.query_index);
    frame1.add_map_match(MapPointMatch{*point, match.train_index});
    frame2.add_map_match(MapPointMatch{*point, match.query_index});

    // Set color
    auto uv = frame1.keypoint(match.train_index).pt;
    auto bgr_color = frame1.image().at<cv::Vec3b>(uv.y, uv.x);
    point->set_color(cv::Vec3b(bgr_color[2], bgr_color[1], bgr_color[0]));

    // Add to map
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

void Map::add_association(Frame& frame, const MapPointMatch& match)
{
    const_cast<MapPoint&>(match.point).add_observation(&frame, match.keypoint_index);
    frame.add_map_match(match);
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