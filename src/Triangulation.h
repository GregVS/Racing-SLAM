#include <Eigen/Dense>
#include <vector>

#include "Camera.h"
#include "FeatureExtractor.h"

namespace slam {

std::pair<std::vector<Eigen::Vector2f>, std::vector<Eigen::Vector2f>>
get_matching_points(const ExtractedFeatures& features1,
                    const ExtractedFeatures& features2,
                    const std::vector<FeatureMatch>& matches);

std::vector<Eigen::Vector3f> triangulate_points(const std::vector<Eigen::Vector2f>& points1,
                                                const std::vector<Eigen::Vector2f>& points2,
                                                const Eigen::Matrix4f& pose1,
                                                const Eigen::Matrix4f& pose2,
                                                const Camera& camera);

} // namespace slam::geometry