#pragma once

#include <optional>

#include "Camera.h"
#include "Map.h"
#include "VideoLoader.h"

namespace slam {

struct SlamConfig {
    bool triangulate_points = true;
    bool bundle_adjust = true;
    bool optimize_pose = true;
    bool cull_points = true;
    bool essential_matrix_estimation = true;
};

class Slam {
  public:
    Slam(const VideoLoader& video_loader,
         const Camera& camera,
         const cv::Mat& image_mask,
         const SlamConfig& config = SlamConfig());

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
    SlamConfig m_config;

    // State
    VideoLoader m_video_loader;
    size_t m_frame_index = 0;
    Map m_map;
    std::vector<std::shared_ptr<Frame>> m_key_frames;
    std::shared_ptr<Frame> m_last_frame;

    // Private methods
    std::optional<Frame> process_next_frame();
    void cull_points();
};

} // namespace slam