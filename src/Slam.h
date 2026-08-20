#pragma once

#include <optional>
#include <string>

#include <Eigen/Dense>

#include "Camera.h"
#include "ImuStream.h"
#include "InertialAlignment.h"
#include "Map.h"
#include "Mapper.h"
#include "Optimization.h"
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
    std::string imu_path; // data.csv path
    // T_SC: p_camera = T_SC * p_sensor.
    Eigen::Matrix4d imu_to_camera = Eigen::Matrix4d::Identity();
    imu::NoiseDensity imu_noise;
    double imu_noise_inflation = 100.0; // Additional sensor noise
    double attitude_error_density = 2.76e-3;
    bool inertial_pose_seed = true; // Seed each frame's pose from inertial data
    std::string vocabulary_path;    // DBoW2 text vocabulary
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

    /** Meters per map unit, zero before alignment */
    double metric_scale() const;

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
    std::optional<imu::Stream> m_imu;
    optimization::InertialInput m_inertial;
    double m_metric_scale = 0.0;
    double m_scale_uncertainty = 0.0;
    Tracker m_tracker;
    Mapper m_mapper;
    FrameDiagnostics m_diagnostics;

    void record_pose(const Frame& frame);

    /** Align map to metric scale using IMU data */
    enum class AlignmentAttempt { NotEnoughSamples, Rejected, Accepted };
    AlignmentAttempt solve_alignment(double spacing, imu::Alignment& alignment, std::vector<size_t>& sampled);
    void apply_scale(float scale);
    void align_to_metric_scale();
};

} // namespace slam
