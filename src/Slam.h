#pragma once

#include <optional>
#include <unordered_map>

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
    std::vector<float> metric_steps;
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
    std::vector<std::shared_ptr<Frame>> m_key_frames;
    std::shared_ptr<Frame> m_last_frame;
    std::vector<Eigen::Matrix4f> m_trajectory;

    struct FeatureTrack {
        std::vector<std::pair<Eigen::Matrix4f, Eigen::Vector2f>> observations;
    };
    std::unordered_map<size_t, FeatureTrack> m_tracks;

    // Private methods
    std::pair<ExtractedFeatures, std::vector<FeatureMatch>> track_features(const cv::Mat& image);
    bool needs_key_frame(const Frame& frame, const Frame& last_key_frame) const;
    void record_pose(const Frame& frame);
    void cull_points();
    std::vector<FeatureMatch> initial_pose_estimate(Frame& frame,
                                                    const std::vector<FeatureMatch>& matches);
    void update_tracks(const std::vector<FeatureMatch>& matches);
    void track_from_last_frame(Frame& frame, const std::vector<FeatureMatch>& matches);
    void triangulate_tracks(Frame& frame);
    float metric_distance(size_t from, size_t to) const;
    void match_with_last_key_frame(Frame& frame);
    void optimize_pose(Frame& frame);
    void match_with_map(Frame& frame);
    void init_key_frame(Frame& frame);
};

} // namespace slam