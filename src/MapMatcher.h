#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "features/FeatureExtractor.h"

namespace slam {

class Camera;
class Frame;
class KeyFrame;
class Map;

class MapMatcher {
  public:
    MapMatcher(const Camera& camera, float max_descriptor_distance, cv::NormTypes norm_type);

    /** Match to every point in the map via reprojection */
    std::vector<MapPointMatch> match_map(const Frame& frame, Map& map) const;

    /** Match to a key frame's points via reprojection */
    std::vector<MapPointMatch> match_key_frame(const Frame& frame, Map& map, KeyFrame* key_frame) const;

    /** Descriptor match to a key frame's points via knn */
    std::vector<MapPointMatch> match_descriptors(const Frame& frame, const KeyFrame& key_frame) const;

  private:
    std::vector<MapPointMatch> match(const Frame& frame, Map& map, KeyFrame* required_observer) const;

    const Camera& m_camera;
    float m_max_descriptor_distance;
    cv::NormTypes m_norm_type;
};

} // namespace slam
