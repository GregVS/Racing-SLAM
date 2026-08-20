#include "LocalWindow.h"

#include <unordered_set>

#include "Frame.h"
#include "MapPoint.h"

namespace slam::optimization {

std::vector<FrameConfig> build_local_window(const std::vector<std::shared_ptr<KeyFrame>>& key_frames,
                                            Frame& new_frame,
                                            size_t window_size,
                                            bool fix_oldest)
{
    size_t first_optimized = key_frames.size() > window_size ? key_frames.size() - window_size : 2;
    std::unordered_set<const Frame*> window{&new_frame};
    for (size_t i = first_optimized; i < key_frames.size(); i++) {
        window.insert(key_frames[i].get());
    }

    std::unordered_set<const Frame*> anchors;
    for (const Frame* window_frame : window) {
        for (const auto& match : window_frame->map_matches()) {
            for (const auto& [observer, _] : match.point.observations()) {
                if (window.find(observer) == window.end()) {
                    anchors.insert(observer);
                }
            }
        }
    }

    std::vector<FrameConfig> frames;
    frames.reserve(key_frames.size() + 1);
    bool included = false;
    for (size_t i = 0; i < key_frames.size(); i++) {
        auto* key_frame = key_frames[i].get();
        bool fixed = i < 2; // for scale anchoring
        if (fix_oldest && i == first_optimized) {
            fixed = true;
        }
        if (window.find(key_frame) != window.end()) {
            frames.push_back({!fixed, key_frame});
            included = included || key_frame == &new_frame;
        } else if (fixed || anchors.find(key_frame) != anchors.end()) {
            frames.push_back({false, key_frame});
        }
    }
    if (!included) {
        frames.push_back({true, &new_frame});
    }
    return frames;
}

} // namespace slam::optimization
