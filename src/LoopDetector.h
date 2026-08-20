#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "Optimization.h"

namespace slam {

class Camera;
class KeyFrame;
struct SlamConfig;

namespace features {
class BaseFeatureExtractor;
}

struct LoopEdge {
    Eigen::Vector3f from = Eigen::Vector3f::Zero();
    Eigen::Vector3f to = Eigen::Vector3f::Zero();
    bool verified = false;
};

struct LoopQueryResult {
    std::vector<LoopEdge> edges;
    size_t candidate_index = 0;
    size_t matches = 0;
    float score = 0.0F;
    bool verified = false;

    // query index > candidate index
    const KeyFrame* query = nullptr;
    const KeyFrame* candidate = nullptr;
    std::vector<Eigen::Vector2f> query_uv;
    std::vector<Eigen::Vector2f> candidate_uv;
};

class LoopDetector {
  public:
    LoopDetector(const SlamConfig& config, const Camera& camera, const features::BaseFeatureExtractor& extractor);
    ~LoopDetector();

    /** Rank old key frames by BoW similarity */
    void query(KeyFrame& key_frame, const std::vector<std::shared_ptr<KeyFrame>>& key_frames);

    const LoopQueryResult& last() const;

    /** True once after a new unique verified loop is stored */
    bool consume_new_loop();
    const std::vector<optimization::PoseGraphConstraint>& constraints() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> p_impl;
};

} // namespace slam
