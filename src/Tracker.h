#pragma once

#include <memory>
#include <opencv2/opencv.hpp>

#include "Camera.h"
#include "Map.h"
#include "MapMatcher.h"
#include "TrackStore.h"
#include "Trajectory.h"

namespace slam {

class Frame;
class KeyFrame;
struct SlamConfig;

/** Follows features from frame to frame and places each new frame against the map */
class Tracker {
  public:
    Tracker(const Camera& camera,
            const cv::Mat& static_mask,
            const features::BaseFeatureExtractor& feature_extractor,
            const SlamConfig& config,
            Map& map);

    /** Builds the next frame, estimates its pose and associates it with map points */
    std::shared_ptr<Frame> track(const cv::Mat& image,
                                 size_t frame_index,
                                 const Trajectory& trajectory,
                                 KeyFrame& last_key_frame,
                                 size_t num_key_frames);

    TrackStore& tracks();
    Frame& last_frame() const;
    bool has_last_frame() const;
    void set_last_frame(const std::shared_ptr<Frame>& frame);

  private:
    std::pair<ExtractedFeatures, std::vector<FeatureMatch>> track_features(const cv::Mat& image);
    std::vector<FeatureMatch> initial_pose_estimate(Frame& frame,
                                                    const std::vector<FeatureMatch>& matches,
                                                    const Trajectory& trajectory,
                                                    size_t num_key_frames);
    void track_from_last_frame(Frame& frame, const std::vector<FeatureMatch>& matches);
    void match_with_last_key_frame(Frame& frame, KeyFrame& last_key_frame);
    void match_with_map(Frame& frame);
    void optimize_pose(Frame& frame);

    const Camera& m_camera;
    const cv::Mat& m_static_mask;
    const features::BaseFeatureExtractor& m_feature_extractor;
    const SlamConfig& m_config;
    Map& m_map;
    MapMatcher m_map_matcher;

    TrackStore m_tracks;
    std::shared_ptr<Frame> m_last_frame;
};

} // namespace slam
