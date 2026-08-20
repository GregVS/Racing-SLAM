#pragma once

#include <memory>
#include <vector>

#include "Optimization.h"

namespace slam {

class Frame;
class KeyFrame;

namespace optimization {

std::vector<FrameConfig> build_local_window(const std::vector<std::shared_ptr<KeyFrame>>& key_frames,
                                            Frame& new_frame,
                                            size_t window_size,
                                            bool fix_oldest = false);

} // namespace optimization
} // namespace slam
