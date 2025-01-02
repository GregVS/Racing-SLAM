#include "Map.h"

namespace slam
{

Map::Map(const Camera &camera)
        : m_camera(camera)
{
}

Frame &Map::add_frame(Frame &&frame)
{
    m_frames.push_back(std::make_unique<Frame>(std::move(frame)));
    return *m_frames.back();
}

const std::vector<std::unique_ptr<Frame> > &Map::get_frames() const
{
    return m_frames;
}

MapPoint &Map::get_map_point(int id)
{
    return m_points.at(id);
}

MapPoint &Map::add_map_point(cv::Point3f position)
{
    int id = m_next_point_id;
    m_points.insert({ id, MapPoint(id, position) });
    m_next_point_id++;
    return m_points.at(id);
}

void Map::remove_map_point(int id)
{
    m_points.erase(id);
}

int Map::get_next_frame_id() const
{
    return m_frames.size();
}

Frame &Map::get_last_frame()
{
    return *m_frames.back();
}

std::unordered_map<int, MapPoint> &Map::get_map_points()
{
    return m_points;
}

const std::unordered_map<int, MapPoint> &Map::get_map_points() const
{
    return m_points;
}

const Camera &Map::get_camera() const
{
    return m_camera;
}

};
