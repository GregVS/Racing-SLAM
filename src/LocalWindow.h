#pragma once

#include <memory>
#include <vector>

#include "Optimization.h"

namespace slam {

class Frame;
class KeyFrame;

namespace optimization {

struct LocalWindow {
    std::vector<FrameConfig> frames;
    std::vector<StepConstraint> step_constraints;
};

LocalWindow build_local_window(const std::vector<std::shared_ptr<KeyFrame>>& key_frames,
                               Frame& new_frame,
                               size_t window_size,
                               const std::vector<float>& metric_steps);

} // namespace optimization
} // namespace slam
