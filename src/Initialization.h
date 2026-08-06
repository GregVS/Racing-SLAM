#pragma once

#include <memory>
#include <opencv2/opencv.hpp>

#include "Camera.h"
#include "Map.h"
#include "VideoLoader.h"
#include "features/FeatureExtractor.h"

namespace slam {

class Frame;
class KeyFrame;
struct SlamConfig;

struct InitializationResult {
    std::shared_ptr<KeyFrame> ref_frame;
    std::shared_ptr<KeyFrame> query_frame;
    size_t next_frame_index = 0;
};

/** Consumes frames until two support a scaled two view reconstruction. Empty frames on failure */
InitializationResult initialize_map(VideoLoader& video_loader,
                                    const Camera& camera,
                                    const cv::Mat& static_mask,
                                    const features::BaseFeatureExtractor& feature_extractor,
                                    const SlamConfig& config,
                                    Map& map);

} // namespace slam
