#pragma once

#include <Eigen/Dense>
#include <memory>
#include <unordered_set>

#include "MapPoint.h"
#include "Features.h"

namespace slam {

class Map {
  public:
    Map();

    void add_point(const Eigen::Vector3f& position);

    void add_point(std::unique_ptr<MapPoint>&& point);

    void create_point(const Eigen::Vector3f& position, Frame& frame1, Frame& frame2, FeatureMatch& match);

    void remove_point(MapPoint* point);

    void add_association(Frame& frame, const MapPointMatch& match);

    // Const iterator for map points
    class const_iterator {
      public:
        const_iterator(std::unordered_set<std::unique_ptr<MapPoint>>::const_iterator it);
        const MapPoint& operator*() const;
        const_iterator& operator++();
        bool operator!=(const const_iterator& other) const;

      private:
        std::unordered_set<std::unique_ptr<MapPoint>>::const_iterator m_it;
    };

    // Mutable iterator for map points
    class iterator {
      public:
        iterator(std::unordered_set<std::unique_ptr<MapPoint>>::iterator it);
        MapPoint& operator*() const;
        iterator& operator++();
        bool operator!=(const iterator& other) const;

      private:
        std::unordered_set<std::unique_ptr<MapPoint>>::iterator m_it;
    };

    iterator begin();
    iterator end();

    const_iterator begin() const;
    const_iterator end() const;

  private:
    std::unordered_set<std::unique_ptr<MapPoint>> m_points;
};

} // namespace slam