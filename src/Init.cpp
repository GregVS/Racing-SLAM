#include "Init.h"

#include "PoseEstimation.h"

namespace slam::init {

InitializerResult find_initializing_frames(std::function<std::shared_ptr<const Frame>()> next_frame,
                                           const Camera& camera)
{
    auto ref_frame = next_frame();
    std::shared_ptr<const Frame> query_frame = nullptr;
    int ref_chances = 0;

    while (query_frame = next_frame()) {
        ref_chances++;

        // Check if we've tried too many times
        if (ref_chances > MAX_REF_CHANCES) {
            std::cout << "Max ref chances reached" << std::endl;
            ref_frame = query_frame;
            ref_chances = 0;
            continue;
        }

        // Check if the reference frame has enough keypoints
        if (ref_frame->features().keypoints.size() < MIN_KEYPOINTS) {
            std::cout << "Ref frame has too few keypoints" << std::endl;
            ref_frame = query_frame;
            ref_chances = 0;
            continue;
        }

        // Check if the query frame has enough keypoints
        if (query_frame->features().keypoints.size() < MIN_KEYPOINTS) {
            std::cout << "Frame has too few keypoints: " << query_frame->features().keypoints.size()
                      << std::endl;
            continue;
        }

        // Match features and check if there are enough matches
        auto matches = features::match_features(ref_frame->features(), query_frame->features());
        if (matches.size() < MIN_MATCHES) {
            std::cout << "Frame has too few matches: " << matches.size() << std::endl;
            continue;
        }

        // Estimate pose and check if there are enough good matches
        auto pose_estimate = pose::estimate_pose(matches,
                                                 ref_frame->features(),
                                                 query_frame->features(),
                                                 camera);
        int good_matches = 0;
        for (int i = 0; i < matches.size(); i++) {
            auto ref_keypoint = ref_frame->keypoint(matches[i].train_index);
            auto curr_keypoint = query_frame->keypoint(matches[i].query_index);
            auto distance = cv::norm(ref_keypoint.pt - curr_keypoint.pt);
            if (distance > GOOD_MATCH_DISTANCE) {
                good_matches++;
            }
        }

        if (good_matches < MIN_GOOD_MATCHES) {
            std::cout << "Frame has too few good matches: " << good_matches << " / "
                      << matches.size() << std::endl;
            continue;
        }

        return InitializerResult{ref_frame,
                                 query_frame,
                                 pose_estimate.inlier_matches,
                                 pose_estimate.pose};
    }

    return InitializerResult{nullptr, nullptr, {}, Eigen::Matrix4f::Identity()};
}

} // namespace slam::init