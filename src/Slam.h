#pragma once

#include "Camera.h"
#include "KeyFrame.h"
#include "Map.h"
#include "VideoLoader.h"

namespace slam {

class Slam {
  public:
    Slam(const VideoLoader& video_loader, const Camera& camera, const cv::Mat& image_mask);

    void initialize();
    void step();

    const std::vector<std::shared_ptr<KeyFrame>>& key_frames() const;
    const Map& map() const;
    const std::shared_ptr<Frame>& frame() const;

  private:
    // Configuration
    Camera m_camera;
    cv::Mat m_static_mask; // Defines the region of interest for feature extraction

    // State
    VideoLoader m_video_loader;
    size_t m_frame_index = 0;
    std::vector<std::shared_ptr<KeyFrame>> m_key_frames;
    Map m_map;
    std::shared_ptr<Frame> m_frame;
    Eigen::Matrix4f m_pose;

    // Private methods
    std::shared_ptr<Frame> process_next_frame();
};

} // namespace slam