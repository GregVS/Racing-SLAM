#pragma once

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <limits>
#include <memory>

#include "Frame.h"
#include "Camera.h"

namespace slam
{

struct MapPoint;
struct Map;

class Map {
public:
    Map(const Camera &camera);

    Frame &add_frame(Frame &&frame);

    Frame &get_last_frame();

    MapPoint &get_map_point(int id);

    MapPoint &add_map_point(cv::Point3f position);

    void remove_map_point(int id);

    const std::vector<std::unique_ptr<Frame>> &get_frames() const;

    int get_next_frame_id() const;

    std::unordered_map<int, MapPoint> &get_map_points();

    const std::unordered_map<int, MapPoint> &get_map_points() const;

    const Camera &get_camera() const;

private:
    std::unordered_map<int, MapPoint> m_points;
    std::vector<std::unique_ptr<Frame>> m_frames;
    Camera m_camera;

    int m_next_point_id = 0;
};

};