#include "Init.h"

#include "PoseEstimation.h"
#include "Triangulation.h"

namespace slam::init {

std::optional<Initialization>
find_initializing_frames(std::function<std::optional<Frame>()> next_frame,
                         const Camera& camera,
                         const features::BaseFeatureExtractor& feature_extractor)
{
    // Get the first frame
    auto ref_frame_opt = next_frame();
    if (!ref_frame_opt) {
        return std::nullopt;
    }
    Frame ref_frame = std::move(*ref_frame_opt);
    int ref_chances = 0;

    while (true) {
        // Get the next frame
        auto query_frame_opt = next_frame();
        if (!query_frame_opt) {
            break;
        }
        Frame query_frame = std::move(*query_frame_opt);
        ref_chances++;

        // After too many attempts, reset ref frame
        if (ref_chances > MAX_REF_CHANCES) {
            std::cout << "Max ref chances reached" << std::endl;
            ref_frame = std::move(query_frame);
            ref_chances = 0;
            continue;
        }

        // Estimate pose
        auto matches = feature_extractor.match_features(ref_frame.features(),
                                                        query_frame.features());
        auto pose_estimate = pose::estimate_pose(ref_frame.features(),
                                                 query_frame.features(),
                                                 matches,
                                                 camera);
        ref_frame.set_pose(Eigen::Matrix4f::Identity());
        query_frame.set_pose(pose_estimate.pose);

        // Triangulate points
        auto triangulated_points = triangulation::triangulate_points(ref_frame,
                                                                     query_frame,
                                                                     matches,
                                                                     camera);
        if (triangulated_points.size() < MIN_TRIANGULATED_POINTS) {
            std::cout << "Frame has too few triangulated points: " << triangulated_points.size()
                      << " / " << matches.size() << std::endl;
            continue;
        }

        return Initialization{std::move(ref_frame), std::move(query_frame), matches};
    }

    return std::nullopt;
}

} // namespace slam::init