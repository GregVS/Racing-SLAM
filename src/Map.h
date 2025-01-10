#pragma once

#include <Eigen/Dense>
#include <memory>
#include <unordered_set>
#include <vector>

namespace slam {

class KeyFrame;

class MapPoint {
  public:
    MapPoint(const Eigen::Vector3f& position);

    const Eigen::Vector3f& position() const;

    void add_observation(const std::shared_ptr<KeyFrame>& key_frame, int index);

    const std::unordered_map<std::shared_ptr<KeyFrame>, int>& observations() const;

  private:
    Eigen::Vector3f m_position;
    std::unordered_map<std::shared_ptr<KeyFrame>, int> m_observations;
};

class Map {
  public:
    Map();

    void add_point(const Eigen::Vector3f& position);

    void add_point(std::unique_ptr<MapPoint>&& point);

    void add_observation(const MapPoint& point, const std::shared_ptr<KeyFrame>& key_frame);

    // Custom iterator for map points with type MapPoint&
    class iterator {
        public:
            iterator(std::unordered_set<std::unique_ptr<MapPoint>>::const_iterator it);
            MapPoint& operator*() const;
            iterator& operator++();
            bool operator!=(const iterator& other) const;
        private:
            std::unordered_set<std::unique_ptr<MapPoint>>::const_iterator m_it;
    };

    iterator begin() const;
    iterator end() const;


  private:
    std::unordered_set<std::unique_ptr<MapPoint>> m_points;
};

} // namespace slam