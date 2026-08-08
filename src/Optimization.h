#pragma once

#include <Eigen/Dense>
#include <utility>
#include <vector>

namespace slam {
// Forward Declarations
class Map;
class MapPoint;
class Frame;
class Camera;
} // namespace slam

namespace slam::optimization {

struct FrameConfig {
    bool optimize;
    Frame* frame;
};

struct OptimizationConfig {
    bool optimize_points;
    const std::vector<FrameConfig> frames;
};

bool optimize(const OptimizationConfig& config, const Camera& camera, Map& map);

/** Holds the state an optimization overwrites so an unhealthy result can be undone */
class Snapshot {
  public:
    Snapshot(const OptimizationConfig& config, Map& map, bool include_points);
    void restore() const;

  private:
    std::vector<std::pair<Frame*, Eigen::Matrix4f>> m_poses;
    std::vector<std::pair<MapPoint*, Eigen::Vector3f>> m_positions;
};

} // namespace slam::optimization
