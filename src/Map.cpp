#include "Map.h"
#include "Frame.h"

namespace slam {

// Map
Map::Map() {}

void Map::add_point(const Eigen::Vector3f& position)
{
    m_points.push_back(std::make_unique<MapPoint>(position));
}

void Map::add_point(std::unique_ptr<MapPoint>&& point)
{
    m_points.push_back(std::move(point));
}

MapPoint& Map::create_point(const Eigen::Vector3f& position,
                            KeyFrame& frame1,
                            KeyFrame& frame2,
                            FeatureMatch& match)
{
    auto point = std::make_unique<MapPoint>(position);
    auto* created = point.get();

    // Support for grayscale images
    auto uv = frame1.keypoint(match.train_index).pt;
    if (frame1.image().channels() == 1) {
        auto gray = frame1.image().at<uchar>(uv.y, uv.x);
        point->set_color(cv::Vec3b(gray, gray, gray));
    } else {
        auto bgr_color = frame1.image().at<cv::Vec3b>(uv.y, uv.x);
        point->set_color(cv::Vec3b(bgr_color[2], bgr_color[1], bgr_color[0]));
    }

    // Add to map
    m_points.push_back(std::move(point));
    associate(frame1, *created, match.train_index);
    associate(frame2, *created, match.query_index);
    return *created;
}

MapPoint& Map::create_point(const Eigen::Vector3f& position, KeyFrame& frame, size_t keypoint_index)
{
    auto point = std::make_unique<MapPoint>(position);
    auto* created = point.get();

    auto uv = frame.keypoint(keypoint_index).pt;
    if (frame.image().channels() == 1) {
        auto gray = frame.image().at<uchar>(uv.y, uv.x);
        created->set_color(cv::Vec3b(gray, gray, gray));
    } else {
        auto bgr_color = frame.image().at<cv::Vec3b>(uv.y, uv.x);
        created->set_color(cv::Vec3b(bgr_color[2], bgr_color[1], bgr_color[0]));
    }

    m_points.push_back(std::move(point));
    associate(frame, *created, keypoint_index);
    return *created;
}

void Map::remove_point(MapPoint* point)
{
    for (auto it = m_points.begin(); it != m_points.end(); ++it) {
        if (it->get() != point) {
            continue;
        }
        const auto observations = (*it)->observations();
        for (const auto& [frame, index] : observations) {
            frame->remove_map_match({**it, index});
        }
        m_points.erase(it);
        break;
    }
}

void Map::fuse(MapPoint& kept, MapPoint& discarded)
{
    if (&kept == &discarded) {
        return;
    }
    const auto observations = discarded.observations();
    for (const auto& [frame, index] : observations) {
        disassociate(*frame, discarded);
        if (kept.is_observed_by(frame) || frame->is_matched(index) || frame->is_matched(kept)) {
            continue;
        }
        associate(*frame, kept, index);
    }
    if (discarded.track_consistent()) {
        kept.set_track_consistent();
    }
    remove_point(&discarded);
}

void Map::associate(KeyFrame& frame, MapPoint& point, size_t keypoint_index)
{
    if (frame.is_matched(keypoint_index) && &frame.map_match(keypoint_index) == &point &&
        point.is_observed_by(&frame)) {
        return;
    }
    if (frame.is_matched(keypoint_index)) {
        MapPoint& existing = frame.map_match(keypoint_index);
        if (&existing != &point) {
            disassociate(frame, existing);
        }
    }
    if (point.is_observed_by(&frame)) {
        disassociate(frame, point);
    }
    point.add_observation(&frame, keypoint_index);
    frame.add_map_match(MapPointMatch{point, keypoint_index});
}

void Map::disassociate(KeyFrame& frame, MapPoint& point)
{
    auto observation = point.observations().find(&frame);
    if (observation == point.observations().end()) {
        return;
    }
    frame.remove_map_match(MapPointMatch{point, observation->second});
    point.remove_observation(&frame);
}

size_t Map::size() const
{
    return m_points.size();
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

Map::const_iterator::const_iterator(std::vector<std::unique_ptr<MapPoint>>::const_iterator it)
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

Map::iterator::iterator(std::vector<std::unique_ptr<MapPoint>>::iterator it) : m_it(it) {}

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