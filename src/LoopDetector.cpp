#include "LoopDetector.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <opencv2/calib3d.hpp>

#include "DBoW2/BowVector.h"
#include "DBoW2/FORB.h"
#include "DBoW2/TemplatedVocabulary.h"

#include "Camera.h"
#include "Frame.h"
#include "Helpers.h"
#include "MapMatcher.h"
#include "MapPoint.h"
#include "Slam.h"
#include "features/FeatureExtractor.h"

namespace slam {

namespace {

using OrbVocabulary = DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>;

constexpr double MIN_LOOP_SECONDS = 10.0;
constexpr size_t MIN_KEYFRAME_GAP = 50;
constexpr size_t TOP_CANDIDATES = 3;
constexpr float PEAK_OVER_MEDIAN = 1.25F;
constexpr float MIN_BOW_SCORE = 0.02F;

constexpr size_t MIN_PNP_CORRESPONDENCES = 12;
constexpr size_t MIN_PNP_INLIERS = 20;
constexpr float MIN_PNP_INLIER_RATIO = 0.35F;
constexpr float MIN_SPREAD_FRAC = 0.25F;
constexpr float PNP_REPROJ_ERROR = 4.0F;

struct Candidate {
    KeyFrame* match = nullptr;
    float score = 0.0F;
    size_t frame_gap = 0;
    size_t index = 0;
};

constexpr size_t MIN_LOOP_SEPARATION = 15;
constexpr size_t MIN_CONSISTENT = 3;
constexpr size_t CONSISTENCY_WINDOW = 15;

struct Verification {
    bool ok = false;
    size_t correspondences = 0;
    size_t inliers = 0;
    float spread = 0.0F;
    float drift = 0.0F;
    float gap = 0.0F;
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    std::vector<Eigen::Vector2f> query_uv;
    std::vector<Eigen::Vector2f> candidate_uv;
};

float percentile(std::vector<float> values, float fraction)
{
    if (values.empty()) {
        return 0.0F;
    }
    std::sort(values.begin(), values.end());
    const size_t index =
        std::min(values.size() - 1, static_cast<size_t>(fraction * static_cast<float>(values.size() - 1)));
    return values[index];
}

std::vector<cv::Mat> descriptor_rows(const cv::Mat& descriptors)
{
    std::vector<cv::Mat> rows;
    rows.reserve(descriptors.rows);
    for (int i = 0; i < descriptors.rows; i++) {
        rows.push_back(descriptors.row(i));
    }
    return rows;
}

DBoW2::BowVector bow_of(const OrbVocabulary& vocabulary, const ExtractedFeatures& features)
{
    DBoW2::BowVector bow;
    if (!features.descriptors.empty()) {
        vocabulary.transform(descriptor_rows(features.descriptors), bow);
    }
    return bow;
}

Eigen::Vector3f camera_center(const Eigen::Matrix4f& pose)
{
    return -pose.block<3, 3>(0, 0).transpose() * pose.block<3, 1>(0, 3);
}

Eigen::Matrix4f pose_from_rt(const cv::Mat& rvec, const cv::Mat& tvec)
{
    cv::Mat rotation;
    cv::Rodrigues(rvec, rotation);
    cv::Mat rotation64;
    cv::Mat translation64;
    rotation.convertTo(rotation64, CV_64F);
    tvec.convertTo(translation64, CV_64F);
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            pose(i, j) = static_cast<float>(rotation64.at<double>(i, j));
        }
        pose(i, 3) = static_cast<float>(translation64.at<double>(i, 0));
    }
    return pose;
}

void set_correspondences(Verification& verification,
                         const KeyFrame& query,
                         const KeyFrame& candidate,
                         const std::vector<MapPointMatch>& matches)
{
    verification.query_uv.clear();
    verification.candidate_uv.clear();
    verification.query_uv.reserve(matches.size());
    verification.candidate_uv.reserve(matches.size());
    for (const auto& match : matches) {
        bool found = false;
        size_t candidate_index = 0;
        for (const auto& [observer, index] : match.point.observations()) {
            if (observer == &candidate) {
                candidate_index = index;
                found = true;
                break;
            }
        }
        if (!found) {
            continue;
        }
        const cv::KeyPoint& query_kp = query.keypoint(match.keypoint_index);
        const cv::KeyPoint& candidate_kp = candidate.keypoint(candidate_index);
        verification.query_uv.emplace_back(query_kp.pt.x, query_kp.pt.y);
        verification.candidate_uv.emplace_back(candidate_kp.pt.x, candidate_kp.pt.y);
    }
}

float keypoint_spread(const std::vector<Eigen::Vector2f>& uv, int image_width)
{
    if (uv.size() < 2 || image_width <= 0) {
        return 0.0F;
    }
    float min_x = uv.front().x();
    float max_x = min_x;
    for (const auto& point : uv) {
        min_x = std::min(min_x, point.x());
        max_x = std::max(max_x, point.x());
    }
    return (max_x - min_x) / static_cast<float>(image_width);
}

void finish_verification(Verification& verification,
                         const KeyFrame& query,
                         const KeyFrame& candidate,
                         const Eigen::Matrix4f& recovered_pose)
{
    verification.spread = keypoint_spread(verification.query_uv, query.image().cols);
    verification.drift = (camera_center(recovered_pose) - query.camera_center()).norm();
    verification.gap = (camera_center(recovered_pose) - candidate.camera_center()).norm();
    verification.pose = recovered_pose;
    const float ratio = verification.correspondences == 0 ? 0.0F
                                                          : static_cast<float>(verification.inliers) /
                                                                static_cast<float>(verification.correspondences);
    verification.ok = verification.inliers >= MIN_PNP_INLIERS && ratio >= MIN_PNP_INLIER_RATIO &&
                      verification.spread >= MIN_SPREAD_FRAC;
}

Verification
verify_pnp(const KeyFrame& query, const KeyFrame& candidate, const Camera& camera, const MapMatcher& matcher)
{
    Verification verification;

    const std::vector<MapPointMatch> correspondences = matcher.match_descriptors(query, candidate);
    verification.correspondences = correspondences.size();
    if (correspondences.size() < MIN_PNP_CORRESPONDENCES) {
        set_correspondences(verification, query, candidate, correspondences);
        return verification;
    }

    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    object_points.reserve(correspondences.size());
    image_points.reserve(correspondences.size());
    for (const auto& match : correspondences) {
        const Eigen::Vector3f& position = match.point.position();
        object_points.emplace_back(position.x(), position.y(), position.z());
        image_points.push_back(query.keypoint(match.keypoint_index).pt);
    }

    cv::Mat intrinsics;
    cv_utils::intrinsic_mat_cv(camera).convertTo(intrinsics, CV_64F);
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat inliers;
    const bool solved = cv::solvePnPRansac(object_points,
                                           image_points,
                                           intrinsics,
                                           cv::Mat(),
                                           rvec,
                                           tvec,
                                           false,
                                           200,
                                           PNP_REPROJ_ERROR,
                                           0.99,
                                           inliers,
                                           cv::SOLVEPNP_EPNP);
    if (!solved || inliers.empty()) {
        set_correspondences(verification, query, candidate, correspondences);
        return verification;
    }

    std::vector<MapPointMatch> inlier_matches;
    inlier_matches.reserve(static_cast<size_t>(inliers.rows));
    for (int i = 0; i < inliers.rows; i++) {
        const int index = inliers.cols == 1 ? inliers.at<int>(i, 0) : inliers.at<int>(0, i);
        inlier_matches.push_back(correspondences[static_cast<size_t>(index)]);
    }
    verification.inliers = inlier_matches.size();
    set_correspondences(verification, query, candidate, inlier_matches);
    finish_verification(verification, query, candidate, pose_from_rt(rvec, tvec));
    return verification;
}

std::vector<Candidate> rank_candidates(const std::vector<Candidate>& considered, size_t query_frame)
{
    std::vector<float> all_scores;
    all_scores.reserve(considered.size());
    for (const auto& candidate : considered) {
        all_scores.push_back(candidate.score);
    }

    const float median_score = percentile(all_scores, 0.5F);
    const float score_thresh = std::max(MIN_BOW_SCORE, median_score * PEAK_OVER_MEDIAN);

    std::vector<Candidate> ranked;
    for (size_t i = 0; i < considered.size(); i++) {
        if (considered[i].score < score_thresh) {
            continue;
        }
        const float left = i == 0 ? 0.0F : considered[i - 1].score;
        const float right = i + 1 == considered.size() ? 0.0F : considered[i + 1].score;
        if (considered[i].score < left || considered[i].score < right) {
            continue;
        }
        ranked.push_back(considered[i]);
    }
    std::sort(ranked.begin(), ranked.end(), [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    if (ranked.size() > TOP_CANDIDATES) {
        ranked.resize(TOP_CANDIDATES);
    }
    if (ranked.empty() && !considered.empty()) {
        const auto best = std::max_element(considered.begin(),
                                           considered.end(),
                                           [](const Candidate& a, const Candidate& b) { return a.score < b.score; });
        std::cout << "Loop rejected: best kf " << best->match->index() << " bow " << best->score << '\n';
    }
    return ranked;
}

size_t best_candidate(const std::vector<Verification>& verifications)
{
    size_t best = 0;
    bool found_verified = false;
    for (size_t i = 0; i < verifications.size(); i++) {
        if (verifications[i].ok && (!found_verified || verifications[i].inliers > verifications[best].inliers)) {
            best = i;
            found_verified = true;
        }
    }
    if (!found_verified) {
        for (size_t i = 1; i < verifications.size(); i++) {
            if (verifications[i].inliers > verifications[best].inliers) {
                best = i;
            }
        }
    }
    return best;
}

LoopQueryResult publish_result(const KeyFrame& query,
                               const std::vector<Candidate>& ranked,
                               const std::vector<Verification>& verifications)
{
    LoopQueryResult result;
    for (size_t i = 0; i < ranked.size(); i++) {
        const auto& candidate = ranked[i];
        const auto& verification = verifications[i];
        result.edges.push_back({query.camera_center(), candidate.match->camera_center(), verification.ok});
    }

    const size_t display = best_candidate(verifications);
    result.candidate_index = ranked[display].match->index();
    result.score = ranked[display].score;
    result.matches = verifications[display].inliers;
    result.verified = verifications[display].ok;
    result.query = &query;
    result.candidate = ranked[display].match;
    result.query_uv = verifications[display].query_uv;
    result.candidate_uv = verifications[display].candidate_uv;
    return result;
}

} // namespace

struct LoopDetector::Impl {
    Impl(const Camera& camera, const features::BaseFeatureExtractor& extractor)
        : camera(camera), matcher(camera, extractor.max_distance(), extractor.norm_type())
    {
    }

    const Camera& camera;
    MapMatcher matcher;
    float seconds_per_frame = 0.0F;
    std::unique_ptr<OrbVocabulary> vocabulary;
    std::vector<DBoW2::BowVector> bows;
    LoopQueryResult last;
    std::vector<optimization::PoseGraphConstraint> constraints;
    bool new_loop = false;

    struct StreakHit {
        size_t query_index = 0;
        size_t candidate_index = 0;
        KeyFrame* candidate = nullptr;
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        size_t inliers = 0;
        float drift = 0.0F;
    };
    std::vector<StreakHit> streak;

    std::vector<Candidate> score_candidates(const KeyFrame& key_frame,
                                            const std::vector<std::shared_ptr<KeyFrame>>& key_frames);
    void update_streak(size_t from,
                       const KeyFrame& key_frame,
                       const std::vector<Candidate>& ranked,
                       const std::vector<Verification>& verifications);
};

std::vector<Candidate> LoopDetector::Impl::score_candidates(const KeyFrame& key_frame,
                                                            const std::vector<std::shared_ptr<KeyFrame>>& key_frames)
{
    bows.resize(key_frames.size());
    const size_t query_index = key_frames.size() - 1;
    if (bows[query_index].empty()) {
        bows[query_index] = bow_of(*vocabulary, key_frame.features());
    }

    std::vector<Candidate> considered;
    considered.reserve(query_index);
    for (size_t i = 0; i < query_index; i++) {
        if (query_index - i < MIN_KEYFRAME_GAP) {
            continue;
        }
        KeyFrame* candidate = key_frames[i].get();
        const double dt = static_cast<double>(key_frame.index() - candidate->index()) * seconds_per_frame;
        if (dt < MIN_LOOP_SECONDS) {
            continue;
        }
        if (bows[i].empty()) {
            bows[i] = bow_of(*vocabulary, candidate->features());
        }
        const auto score = static_cast<float>(vocabulary->score(bows[query_index], bows[i]));
        considered.push_back({candidate, score, key_frame.index() - candidate->index(), i});
    }
    return considered;
}

void LoopDetector::Impl::update_streak(size_t from,
                                       const KeyFrame& key_frame,
                                       const std::vector<Candidate>& ranked,
                                       const std::vector<Verification>& verifications)
{
    const size_t seed = best_candidate(verifications);
    if (!verifications[seed].ok) {
        if (!streak.empty()) {
            std::cout << "Loop streak reset at kf " << key_frame.index() << '\n';
        }
        streak.clear();
        return;
    }

    size_t chosen = seed;
    // Try to find best candidate that continues the streak, otherwise reset it
    const bool consecutive = !streak.empty() && from == streak.back().query_index + 1;
    if (consecutive) {
        size_t continued = ranked.size();
        for (size_t i = 0; i < ranked.size(); i++) {
            const size_t index_gap = ranked[i].index > streak.back().candidate_index
                                         ? ranked[i].index - streak.back().candidate_index
                                         : streak.back().candidate_index - ranked[i].index;
            if (!verifications[i].ok || index_gap > CONSISTENCY_WINDOW) {
                continue;
            }
            if (continued == ranked.size() || verifications[i].inliers > verifications[continued].inliers) {
                continued = i;
            }
        }
        if (continued < ranked.size()) {
            chosen = continued;
        } else {
            streak.clear();
        }
    } else {
        streak.clear();
    }

    const auto& candidate = ranked[chosen];
    const auto& verification = verifications[chosen];
    streak.push_back(
        {from, candidate.index, candidate.match, verification.pose, verification.inliers, verification.drift});

    if (streak.size() < MIN_CONSISTENT) {
        return;
    }

    const auto& hit = streak.back();
    for (const auto& constraint : constraints) {
        const size_t from_gap = from > constraint.from ? from - constraint.from : constraint.from - from;
        const size_t to_gap = hit.candidate_index > constraint.to ? hit.candidate_index - constraint.to
                                                                  : constraint.to - hit.candidate_index;
        if (from_gap < MIN_LOOP_SEPARATION && to_gap < MIN_LOOP_SEPARATION) {
            return;
        }
    }
    const Eigen::Matrix4d relative = hit.pose.cast<double>() * hit.candidate->pose().inverse().cast<double>();
    constraints.push_back({from, hit.candidate_index, relative});
    new_loop = true;
}

LoopDetector::LoopDetector(const SlamConfig& config,
                           const Camera& camera,
                           const features::BaseFeatureExtractor& extractor)
    : p_impl(std::make_unique<Impl>(camera, extractor))
{
    p_impl->seconds_per_frame = config.seconds_per_frame;
    if (config.vocabulary_path.empty()) {
        std::cout << "No BoW vocabulary path for loop detection\n";
        return;
    }
    auto vocabulary = std::make_unique<OrbVocabulary>();
    if (!vocabulary->loadFromTextFile(config.vocabulary_path)) {
        std::cout << "Failed to load BoW vocabulary\n";
        return;
    }
    std::cout << "Loaded BoW vocabulary with " << vocabulary->size() << " words\n";
    p_impl->vocabulary = std::move(vocabulary);
}

LoopDetector::~LoopDetector() = default;

const LoopQueryResult& LoopDetector::last() const
{
    return p_impl->last;
}

bool LoopDetector::consume_new_loop()
{
    const bool added = p_impl->new_loop;
    p_impl->new_loop = false;
    return added;
}

const std::vector<optimization::PoseGraphConstraint>& LoopDetector::constraints() const
{
    return p_impl->constraints;
}

void LoopDetector::query(KeyFrame& key_frame, const std::vector<std::shared_ptr<KeyFrame>>& key_frames)
{
    if (!p_impl->vocabulary) {
        return;
    }
    if (key_frames.size() < MIN_KEYFRAME_GAP + 1) {
        return;
    }

    std::vector<Candidate> ranked;
    time_it("Loop retrieval",
            [&]() { ranked = rank_candidates(p_impl->score_candidates(key_frame, key_frames), key_frame.index()); });

    if (ranked.empty()) {
        p_impl->streak.clear();
        p_impl->last = {};
        return;
    }

    std::vector<Verification> verifications(ranked.size());
    time_it("Loop verify", [&]() {
        for (size_t i = 0; i < ranked.size(); i++) {
            verifications[i] = verify_pnp(key_frame, *ranked[i].match, p_impl->camera, p_impl->matcher);
        }
    });

    p_impl->last = publish_result(key_frame, ranked, verifications);
    p_impl->update_streak(key_frames.size() - 1, key_frame, ranked, verifications);
}

} // namespace slam
