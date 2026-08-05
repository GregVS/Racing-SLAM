#include "YawEstimation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <Eigen/Geometry>

namespace slam::yaw {
namespace {

constexpr float GRID_STEP_RADIANS = 0.1f * 3.14159265358979323846f / 180.0f;
constexpr float INLIER_THRESHOLD_PIXELS = 2.0f;
constexpr float TRUNCATION_THRESHOLD_PIXELS = 4.0f;
constexpr size_t MIN_INLIERS = 30;
constexpr float MIN_INLIER_RATIO = 0.3f;
constexpr float MIN_IMAGE_COVERAGE = 0.5f;
constexpr int GRID_COLUMNS = 4;
constexpr int GRID_ROWS = 3;

struct CandidateScore {
    float cost = std::numeric_limits<float>::infinity();
    size_t inliers = 0;
    std::array<bool, GRID_COLUMNS * GRID_ROWS> occupied{};
};

Eigen::Matrix3f skew(const Eigen::Vector3f& vector)
{
    Eigen::Matrix3f result;
    result << 0.0f, -vector.z(), vector.y(), vector.z(), 0.0f, -vector.x(), -vector.y(), vector.x(), 0.0f;
    return result;
}

CandidateScore score(float yaw,
                     const ExtractedFeatures& previous,
                     const ExtractedFeatures& current,
                     const std::vector<FeatureMatch>& matches,
                     const Camera& camera)
{
    Eigen::Matrix3f rotation = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitY()).toRotationMatrix();
    Eigen::Vector3f forward = (Eigen::Vector3f::UnitZ() + rotation.transpose() * Eigen::Vector3f::UnitZ()).normalized();
    Eigen::Vector3f translation = -rotation * forward;
    Eigen::Matrix3f essential = skew(translation) * rotation;

    const auto& intrinsics = camera.get_intrinsic_matrix();
    float fx = intrinsics(0, 0);
    float fy = intrinsics(1, 1);
    float cx = intrinsics(0, 2);
    float cy = intrinsics(1, 2);
    float focal = std::sqrt(fx * fy);
    float truncation_squared = TRUNCATION_THRESHOLD_PIXELS * TRUNCATION_THRESHOLD_PIXELS;

    CandidateScore result;
    result.cost = 0.0f;
    for (const auto& match : matches) {
        const auto& from = previous.keypoints[match.train_index].pt;
        const auto& to = current.keypoints[match.query_index].pt;
        Eigen::Vector3f x1((from.x - cx) / fx, (from.y - cy) / fy, 1.0f);
        Eigen::Vector3f x2((to.x - cx) / fx, (to.y - cy) / fy, 1.0f);
        Eigen::Vector3f line2 = essential * x1;
        Eigen::Vector3f line1 = essential.transpose() * x2;
        float denominator =
            std::sqrt(line1.x() * line1.x() + line1.y() * line1.y() + line2.x() * line2.x() + line2.y() * line2.y());
        if (denominator < 1e-9f) {
            result.cost += truncation_squared;
            continue;
        }
        float error = std::abs(x2.dot(essential * x1)) / denominator * focal;
        result.cost += std::min(error * error, truncation_squared);
        if (error <= INLIER_THRESHOLD_PIXELS) {
            result.inliers++;
            int column = std::clamp(int(to.x * GRID_COLUMNS / camera.get_width()), 0, GRID_COLUMNS - 1);
            int row = std::clamp(int(to.y * GRID_ROWS / camera.get_height()), 0, GRID_ROWS - 1);
            result.occupied[row * GRID_COLUMNS + column] = true;
        }
    }
    result.cost /= std::max<size_t>(matches.size(), 1);
    return result;
}

} // namespace

YawEstimate estimate(const ExtractedFeatures& previous,
                     const ExtractedFeatures& current,
                     const std::vector<FeatureMatch>& matches,
                     const Camera& camera,
                     float max_yaw_radians)
{
    YawEstimate result;
    if (matches.size() < MIN_INLIERS || max_yaw_radians <= 0.0f) {
        return result;
    }

    float best_yaw = 0.0f;
    CandidateScore best;
    int samples = std::max(2, int(std::ceil(2.0f * max_yaw_radians / GRID_STEP_RADIANS)));
    for (int i = 0; i <= samples; i++) {
        float candidate = -max_yaw_radians + 2.0f * max_yaw_radians * i / samples;
        auto candidate_score = score(candidate, previous, current, matches, camera);
        if (candidate_score.cost < best.cost) {
            best_yaw = candidate;
            best = candidate_score;
        }
    }

    float refinement_step = 2.0f * max_yaw_radians / samples;
    for (int iteration = 0; iteration < 4; iteration++) {
        refinement_step *= 0.2f;
        for (int direction : {-1, 1}) {
            float candidate = std::clamp(best_yaw + direction * refinement_step, -max_yaw_radians, max_yaw_radians);
            auto candidate_score = score(candidate, previous, current, matches, camera);
            if (candidate_score.cost < best.cost) {
                best_yaw = candidate;
                best = candidate_score;
            }
        }
    }

    size_t occupied = std::count(best.occupied.begin(), best.occupied.end(), true);
    result.radians = best_yaw;
    result.inliers = best.inliers;
    result.inlier_ratio = float(best.inliers) / matches.size();
    result.image_coverage = float(occupied) / best.occupied.size();
    result.robust_cost = best.cost;
    result.healthy = result.inliers >= MIN_INLIERS && result.inlier_ratio >= MIN_INLIER_RATIO &&
                     result.image_coverage >= MIN_IMAGE_COVERAGE;
    return result;
}

} // namespace slam::yaw
