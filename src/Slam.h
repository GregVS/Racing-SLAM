#pragma once

#include <optional>

#include "Camera.h"
#include "Map.h"
#include "VideoLoader.h"

namespace slam {

class Slam {
  public:
    Slam(const VideoLoader& video_loader, const Camera& camera, const cv::Mat& image_mask);

    void initialize();
    void step();

    float reprojection_error() const;
    const Map& map() const;
    const Frame& frame() const;
    std::vector<Eigen::Matrix4f> poses() const;

  private:
    // Configuration
    Camera m_camera;
    cv::Mat m_static_mask; // Defines the region of interest for feature extraction

    // State
    VideoLoader m_video_loader;
    size_t m_frame_index = 0;
    Map m_map;
    std::vector<std::shared_ptr<Frame>> m_key_frames;
    std::shared_ptr<Frame> m_frame;

    // Private methods
    std::optional<Frame> process_next_frame();
};

} // namespace slam