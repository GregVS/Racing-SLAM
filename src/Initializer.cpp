#include "Initializer.h"

namespace slam {

Initializer::Initializer(const Camera& camera) : m_camera(camera) {}

void Initializer::reset_ref(const std::shared_ptr<Frame>& frame)
{
    m_ref_frame = frame;
    m_ref_chances = 0;
}

void Initializer::increment_ref_chances(const std::shared_ptr<Frame>& frame)
{
    m_ref_chances++;
    if (m_ref_chances > MAX_REF_CHANCES) {
        reset_ref(frame);
    }
}

std::optional<InitializerResult> Initializer::try_initialize(std::shared_ptr<Frame> frame)
{
    // Check the frame has enough keypoints
    if (frame->features().keypoints.size() < MIN_KEYPOINTS) {
        std::cout << "Frame has too few keypoints: " << frame->features().keypoints.size() << std::endl;
        return std::nullopt;
    } else if (!m_ref_frame || m_ref_frame->features().keypoints.size() < MIN_KEYPOINTS) {
        std::cout << "Ref frame has too few keypoints" << std::endl;
        reset_ref(frame);
        return std::nullopt;
    }

    // Match features and check if there are enough matches
    auto matches = m_feature_extractor.match_features(m_ref_frame->features(), frame->features());
    if (matches.size() < MIN_MATCHES) {
        std::cout << "Frame has too few matches: " << matches.size() << std::endl;
        increment_ref_chances(frame);
        return std::nullopt;
    }

    // Filter matches and check if there are enough filtered matches
    auto [filtered_matches, essential_matrix] =
        m_feature_extractor.filter_matches(matches,
                                           m_ref_frame->features(),
                                           frame->features(),
                                           m_camera);

    if (filtered_matches.size() < MIN_FILTERED_MATCHES) {
        std::cout << "Frame has too few filtered matches: " << filtered_matches.size() << std::endl;
        increment_ref_chances(frame);
        return std::nullopt;
    }

    // Ensure that there is enough distance between the matches
    int good_matches = 0;
    for (const auto& match : filtered_matches) {
        auto ref_keypoint = m_ref_frame->features().keypoints.at(match.train_index);
        auto curr_keypoint = frame->features().keypoints.at(match.query_index);
        auto distance = cv::norm(ref_keypoint.pt - curr_keypoint.pt);
        if (distance > GOOD_MATCH_DISTANCE) {
            good_matches++;
        }
    }
    if ((float)good_matches / filtered_matches.size() < GOOD_MATCH_RATIO) {
        std::cout << "Frame has too few good matches: " << good_matches << " / " << filtered_matches.size() << std::endl;
        increment_ref_chances(frame);
        return std::nullopt;
    }

    return InitializerResult{essential_matrix, filtered_matches};
}

const std::shared_ptr<Frame>& Initializer::ref_frame() const { return m_ref_frame; }

} // namespace slam