#pragma once

#include <memory>
#include <optional>
#include <Eigen/Dense>

#include "FeatureExtractor.h"
#include "Frame.h"
#include "PoseEstimator.h"

namespace slam {

struct InitializerResult {
    std::vector<FeatureMatch> inlier_matches;
    Eigen::Matrix4f pose;
};

class Initializer {
  public:
    Initializer(const Camera& camera);

    std::optional<InitializerResult> try_initialize(std::shared_ptr<Frame> frame);

    const std::shared_ptr<Frame>& ref_frame() const;

    void increment_ref_chances(const std::shared_ptr<Frame>& frame);

  private:
    void reset_ref(const std::shared_ptr<Frame>& frame);

    const FeatureExtractor m_feature_extractor;
    const PoseEstimator m_pose_estimator;
    const Camera m_camera;
    std::shared_ptr<Frame> m_ref_frame;

    int m_ref_chances = 0;

    static constexpr int MIN_KEYPOINTS = 100;
    static constexpr int MIN_MATCHES = 100;
    static constexpr int MAX_REF_CHANCES = 5;
    static constexpr float GOOD_MATCH_DISTANCE = 15.0;
    static constexpr int MIN_GOOD_MATCHES = 50;
};

} // namespace slam