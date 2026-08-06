#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "features/FeatureExtractor.h"

namespace slam {

class Camera;
class Frame;
class KeyFrame;
class Map;

/** Matches keypoints to map points by projection and descriptor distance */
class MapMatcher {
  public:
    MapMatcher(const Camera& camera, float max_descriptor_distance, cv::NormTypes norm_type);

    /** Considers every point in the map */
    std::vector<MapPointMatch> match_map(const Frame& frame, Map& map) const;

    /** Considers only the points a given key frame observes */
    std::vector<MapPointMatch> match_key_frame(const Frame& frame, Map& map, KeyFrame* key_frame) const;

  private:
    std::vector<MapPointMatch> match(const Frame& frame, Map& map, KeyFrame* required_observer) const;

    const Camera& m_camera;
    float m_max_descriptor_distance;
    cv::NormTypes m_norm_type;
};

} // namespace slam
