#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "MapPoint.h"
#include "features/FeatureExtractor.h"

namespace slam {

class Map {
  public:
    Map();

    void add_point(const Eigen::Vector3f& position);

    void add_point(std::unique_ptr<MapPoint>&& point);

    MapPoint& create_point(const Eigen::Vector3f& position, KeyFrame& frame1, KeyFrame& frame2, FeatureMatch& match);

    /** Creates a point seen so far by one frame only */
    MapPoint& create_point(const Eigen::Vector3f& position, KeyFrame& key_frame, size_t keypoint_index);

    void remove_point(MapPoint* point);

    /** Move discarded observations onto kept and delete discarded */
    void fuse(MapPoint& kept, MapPoint& discarded);

    void associate(KeyFrame& key_frame, MapPoint& point, size_t keypoint_index);
    void disassociate(KeyFrame& key_frame, MapPoint& point);

    size_t size() const;

    // Const iterator for map points
    class const_iterator {
      public:
        const_iterator(std::vector<std::unique_ptr<MapPoint>>::const_iterator it);
        const MapPoint& operator*() const;
        const_iterator& operator++();
        bool operator!=(const const_iterator& other) const;

      private:
        std::vector<std::unique_ptr<MapPoint>>::const_iterator m_it;
    };

    // Mutable iterator for map points
    class iterator {
      public:
        iterator(std::vector<std::unique_ptr<MapPoint>>::iterator it);
        MapPoint& operator*() const;
        iterator& operator++();
        bool operator!=(const iterator& other) const;

      private:
        std::vector<std::unique_ptr<MapPoint>>::iterator m_it;
    };

    iterator begin();
    iterator end();

    const_iterator begin() const;
    const_iterator end() const;

  private:
    std::vector<std::unique_ptr<MapPoint>> m_points;
};

} // namespace slam