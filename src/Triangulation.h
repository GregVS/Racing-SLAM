#include <Eigen/Dense>
#include <vector>

#include "FeatureExtractor.h"
#include "Camera.h"

namespace slam {

std::vector<Eigen::Vector3f> triangulate_features(
    const ExtractedFeatures& features1,
    const ExtractedFeatures& features2,
    const std::vector<FeatureMatch>& matches,
    const Eigen::Matrix4f& pose1,
    const Eigen::Matrix4f& pose2,
    const Camera& camera);

} // namespace slam