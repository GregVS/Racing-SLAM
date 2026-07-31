#pragma once

#include <Eigen/Dense>
#include <vector>

#include "Camera.h"
#include "features/FeatureExtractor.h"

namespace slam::triangulation {

// Minimum angle between the two viewing rays, as a cosine
static const float MIN_PARALLAX_COSINE = 0.9999f;

struct TriangulatedPoint {
    Eigen::Vector3f position;
    int match_index;
};

std::pair<std::vector<Eigen::Vector2f>, std::vector<Eigen::Vector2f>>
get_matching_points(const ExtractedFeatures& features1,
                    const ExtractedFeatures& features2,
                    const std::vector<FeatureMatch>& matches);

std::vector<TriangulatedPoint> triangulate_points(const std::vector<Eigen::Vector2f>& points1,
                                                  const std::vector<Eigen::Vector2f>& points2,
                                                  const Eigen::Matrix4f& pose1,
                                                  const Eigen::Matrix4f& pose2,
                                                  const Camera& camera,
                                                  float min_parallax_cosine = MIN_PARALLAX_COSINE,
                                                  float max_reprojection_error = 2.0f);

std::vector<TriangulatedPoint> triangulate_points(const Frame& frame1,
                                                  const Frame& frame2,
                                                  const std::vector<FeatureMatch>& matches,
                                                  const Camera& camera);
;
} // namespace slam::triangulation