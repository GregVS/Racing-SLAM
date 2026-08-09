#pragma once

#include <Eigen/Dense>
#include <vector>

namespace slam {
// Forward Declarations
class Map;
class Frame;
class Camera;
} // namespace slam

namespace slam::optimization {

struct FrameConfig {
    bool optimize;
    Frame* frame;
};

bool refine_pose(Frame& frame, const Camera& camera);

bool bundle_adjust(const std::vector<FrameConfig>& frames, const Camera& camera, Map& map);

} // namespace slam::optimization
