#pragma once

#include <optional>
#include <unordered_map>

#include "Camera.h"
#include "Map.h"
#include "Mapper.h"
#include "Tracker.h"
#include "Trajectory.h"
#include "VideoLoader.h"

namespace slam {

struct SlamConfig {
    bool triangulate_points = true;
    bool bundle_adjust = true;
    bool optimize_pose = true;
    bool cull_points = true;
    bool essential_matrix_estimation = true;
    float seconds_per_frame = 0.0F;
};

/** For visualization purposes */
struct FrameDiagnostics {
    size_t map_size = 0;
    size_t triangulated = 0;
    size_t track_consistent = 0;
    size_t poisoned = 0;
    std::vector<Eigen::Vector3f> culled;
};

class Slam {
  public:
    Slam(const VideoLoader& video_loader,
         const Camera& camera,
         const cv::Mat& image_mask,
         std::unique_ptr<features::BaseFeatureExtractor> feature_extractor,
         const SlamConfig& config = SlamConfig());

    void initialize();

    /** Processes the next frame. Returns false once the video is exhausted. */
    bool step();

    float reprojection_error() const;
    const FrameDiagnostics& diagnostics() const;
    const Map& map() const;
    const Frame& frame() const;
    std::vector<Eigen::Matrix4f> poses() const;

    /** One pose per frame index, including non key frames. Key frames report their current,
     * bundle adjusted pose */
    std::vector<Eigen::Matrix4f> trajectory() const;

  private:
    // Configuration
    Camera m_camera;
    cv::Mat m_static_mask; // Defines the region of interest for feature extraction
    SlamConfig m_config;
    std::unique_ptr<features::BaseFeatureExtractor> m_feature_extractor;

    // State
    VideoLoader m_video_loader;
    size_t m_frame_index = 0;
    Map m_map;
    Trajectory m_trajectory;
    Tracker m_tracker;
    Mapper m_mapper;
    FrameDiagnostics m_diagnostics;

    void record_pose(const Frame& frame);
};

} // namespace slam
