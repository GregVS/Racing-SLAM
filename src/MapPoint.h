#pragma once

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <functional>

namespace slam
{

class Frame;

class MapPoint
{
public:
    MapPoint(int id, cv::Point3f position);

    void add_observation(Frame *frame, int keypointIndex);

    int get_id() const;

    const cv::Point3f &get_position() const;

    void set_position(cv::Point3f position);

    bool is_observed_by(Frame *frame) const;

    using ObservationData = std::pair<Frame*, int>;

    std::vector<ObservationData> get_observations_vector() const;
    
    void for_each_observation(const std::function<void(Frame*, int)>& callback) const;

    size_t observation_count() const;

private:
    const int m_id;
    cv::Point3f m_position;
    std::unordered_map<Frame *, int> m_observations;
};

float map_point_orb_distance(const MapPoint &map_point, const cv::Mat &descriptor);

};