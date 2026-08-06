#include "LocalWindow.h"

#include <unordered_set>

#include "Frame.h"
#include "MapPoint.h"
#include "MotionModel.h"

namespace slam::optimization {

LocalWindow build_local_window(const std::vector<std::shared_ptr<KeyFrame>>& key_frames,
                               Frame& new_frame,
                               size_t window_size,
                               const std::vector<float>& metric_steps)
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

    LocalWindow local_window;
    local_window.frames.reserve(key_frames.size() + 1);
    for (size_t i = 0; i < key_frames.size(); i++) {
        auto* key_frame = key_frames[i].get();
        bool fixed = i < 2; // for scale anchoring
        if (window.find(key_frame) != window.end()) {
            local_window.frames.push_back({!fixed, key_frame, metric_steps.empty()});
        } else if (fixed || anchors.find(key_frame) != anchors.end()) {
            local_window.frames.push_back({false, key_frame});
        }
    }
    local_window.frames.push_back({true, &new_frame, metric_steps.empty()});

    local_window.step_constraints.reserve(local_window.frames.size());
    if (!metric_steps.empty()) {
        std::unordered_set<const Frame*> selected;
        for (const auto& frame_config : local_window.frames) {
            selected.insert(frame_config.frame);
        }
        for (size_t i = 1; i < key_frames.size(); i++) {
            const Frame* previous = key_frames[i - 1].get();
            const Frame* current = key_frames[i].get();
            if (selected.count(previous) && selected.count(current)) {
                local_window.step_constraints.push_back(
                    {previous, current, motion::metric_distance(metric_steps, previous->index(), current->index())});
            }
        }
        local_window.step_constraints.push_back(
            {key_frames.back().get(),
             &new_frame,
             motion::metric_distance(metric_steps, key_frames.back()->index(), new_frame.index())});
    }
    return local_window;
}

} // namespace slam::optimization
