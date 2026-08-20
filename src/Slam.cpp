#include "Slam.h"

#include <algorithm>
#include <cmath>

#include "Frame.h"
#include "Helpers.h"
#include "InertialAlignment.h"
#include "Initialization.h"
#include "features/FeatureExtractor.h"

namespace slam {

namespace {

constexpr size_t MAX_TRACK_SIGHTINGS = 100;

// Alignment parameters
constexpr double ALIGNMENT_SAMPLE_SECONDS[] = {1.0, 0.5}; // Spacings to try in order
constexpr size_t ALIGNMENT_SAMPLES = 12;
constexpr size_t MIN_ALIGNMENT_SAMPLES = ALIGNMENT_SAMPLES;
constexpr double MAX_GRAVITY_MAGNITUDE_ERROR = 1.0;
constexpr double MAX_ALIGNMENT_RESIDUAL = 2.0;
constexpr double MAX_GRAVITY_DIRECTION_UNCERTAINTY = 0.05; // Radians
constexpr double MAX_SCALE_UNCERTAINTY = 0.05;             // Expressed as fraction
constexpr double REFINEMENT_UNCERTAINTY_RATIO = 0.5;       // When scale should be re-aligned

} // namespace

Slam::Slam(const VideoLoader& video_loader,
           const Camera& camera,
           const cv::Mat& image_mask,
           std::unique_ptr<features::BaseFeatureExtractor> feature_extractor,
           const SlamConfig& config)
    : m_video_loader(video_loader), m_camera(camera), m_static_mask(image_mask),
      m_feature_extractor(std::move(feature_extractor)), m_config(config),
      m_tracker(m_camera, m_static_mask, *m_feature_extractor, m_config, m_map),
      m_mapper(m_camera, m_config, m_map, m_inertial), m_loop_detector(m_config, m_camera, *m_feature_extractor)
{
    m_inertial.seconds_per_frame = m_config.seconds_per_frame;
    if (m_config.imu_path.empty()) {
        return;
    }
    m_imu = imu::Stream::load(m_config.imu_path, m_config.imu_to_camera);
    std::cout << "Loaded " << m_imu->size() << " imu samples spanning " << m_imu->first() << " to " << m_imu->last()
              << " s from " << m_config.imu_path << '\n';

    imu::NoiseDensity noise = m_config.imu_noise;
    if (noise.gyro <= 0.0 || noise.accel <= 0.0) {
        std::cout << "Config reports no noise; weighting with the modelled part instead\n";
        noise = imu::NoiseDensity{};
    }

    // Add sensor noise that isn't captured in the spec
    const double inflation = m_config.imu_noise_inflation;
    noise.gyro *= inflation;
    noise.accel *= inflation;
    noise.gyro_bias *= inflation;
    noise.accel_bias *= inflation;
    m_inertial.noise = noise;
    m_inertial.attitude_error_density = m_config.attitude_error_density;

    m_tracker.set_inertial(&*m_imu);
    m_tracker.set_inertial_input(&m_inertial);
}

double Slam::metric_scale() const
{
    return m_metric_scale;
}

Slam::AlignmentAttempt Slam::solve_alignment(double spacing, imu::Alignment& alignment, std::vector<size_t>& sampled)
{
    const auto& key_frames = m_mapper.key_frames();
    sampled.assign(1, key_frames.size() - 1);
    double last_time = m_inertial.time_of(key_frames.back()->index());
    for (size_t i = key_frames.size() - 1; i-- > 0 && sampled.size() < ALIGNMENT_SAMPLES;) {
        const double at = m_inertial.time_of(key_frames[i]->index());
        if (last_time - at >= spacing) {
            sampled.push_back(i);
            last_time = at;
        }
    }
    if (sampled.size() < MIN_ALIGNMENT_SAMPLES) {
        return AlignmentAttempt::NotEnoughSamples;
    }
    std::reverse(sampled.begin(), sampled.end());

    std::vector<Eigen::Matrix3d> rotations;
    std::vector<Eigen::Vector3d> positions;
    std::vector<double> times;
    std::vector<imu::Preintegrated> summaries;
    for (const size_t i : sampled) {
        const imu::State state = inertial_state(*key_frames[i]);
        rotations.push_back(state.rotation);
        positions.push_back(state.position);
        times.push_back(m_inertial.time_of(key_frames[i]->index()));
        if (times.size() > 1) {
            summaries.push_back(
                imu::preintegrate(m_imu->between(times[times.size() - 2], times.back()), m_inertial.noise));
        }
    }

    alignment = imu::align(rotations, positions, times, summaries);
    std::cout << "Inertial alignment at " << spacing << " s spacing over " << rotations.size()
              << " key frames spanning " << times.back() - times.front() << " s: scale " << alignment.scale
              << " m per unit, gravity [" << alignment.gravity.transpose() << "] magnitude " << alignment.gravity.norm()
              << ", residual " << alignment.residual << " m over " << alignment.triples
              << " triples, scale uncertainty " << 100.0 * alignment.scale_uncertainty
              << " %, gravity direction uncertainty " << alignment.gravity_uncertainty * 180.0 / M_PI << " deg\n";

    if (!alignment.valid) {
        std::cout << "Rejected: the solve did not produce a usable scale\n";
        return AlignmentAttempt::Rejected;
    }
    if (std::abs(alignment.gravity_magnitude_error) > MAX_GRAVITY_MAGNITUDE_ERROR) {
        std::cout << "Rejected: recovered gravity is not plausible\n";
        return AlignmentAttempt::Rejected;
    }
    if (alignment.gravity_uncertainty > MAX_GRAVITY_DIRECTION_UNCERTAINTY) {
        std::cout << "Rejected: the motion so far does not determine which way is down\n";
        return AlignmentAttempt::Rejected;
    }
    if (alignment.residual > MAX_ALIGNMENT_RESIDUAL) {
        std::cout << "Rejected: no single scale fits this map\n";
        return AlignmentAttempt::Rejected;
    }
    if (alignment.scale_uncertainty > MAX_SCALE_UNCERTAINTY) {
        std::cout << "Rejected: the motion so far does not determine scale\n";
        return AlignmentAttempt::Rejected;
    }
    return AlignmentAttempt::Accepted;
}

void Slam::apply_scale(float scale)
{
    for (auto& point : m_map) {
        point.set_position(point.position() * scale);
    }
    for (Eigen::Vector3f& position : m_diagnostics.culled) {
        position *= scale;
    }
    for (const auto& key_frame : m_mapper.key_frames()) {
        Eigen::Matrix4f pose = key_frame->pose();
        pose.block<3, 1>(0, 3) *= scale;
        key_frame->set_pose(pose);
    }
    m_trajectory.rescale(scale);
}

void Slam::align_to_metric_scale()
{
    const auto& key_frames = m_mapper.key_frames();
    if (!m_imu || key_frames.empty()) {
        return;
    }

    imu::Alignment alignment;
    std::vector<size_t> sampled;
    double chosen_spacing = 0.0;
    for (const double spacing : ALIGNMENT_SAMPLE_SECONDS) {
        const AlignmentAttempt attempt = solve_alignment(spacing, alignment, sampled);
        if (attempt == AlignmentAttempt::NotEnoughSamples) {
            continue;
        }
        if (attempt == AlignmentAttempt::Accepted) {
            chosen_spacing = spacing;
            break;
        }
    }
    if (chosen_spacing == 0.0) {
        return;
    }
    if (m_inertial.aligned() && alignment.scale_uncertainty > REFINEMENT_UNCERTAINTY_RATIO * m_scale_uncertainty) {
        return;
    }
    std::cout << "Alignment accepted at " << chosen_spacing << " s spacing, scale " << alignment.scale
              << ", uncertainty " << 100.0 * alignment.scale_uncertainty << " % against " << 100.0 * m_scale_uncertainty
              << " % in force\n";

    apply_scale(static_cast<float>(alignment.scale));

    std::vector<bool> from_solve(key_frames.size(), false);
    for (size_t k = 0; k < sampled.size(); k++) {
        InertialState state = key_frames[sampled[k]]->inertial();
        state.velocity = alignment.velocities[k];
        key_frames[sampled[k]]->set_inertial(state);
        from_solve[sampled[k]] = true;
    }

    // Set velocities for key frames that were not in alignment samples
    for (size_t i = key_frames.size() - 1; i-- > 0;) {
        if (from_solve[i]) {
            continue;
        }
        const double from = m_inertial.time_of(key_frames[i]->index());
        const double to = m_inertial.time_of(key_frames[i + 1]->index());
        const imu::Preintegrated summary = imu::preintegrate(m_imu->between(from, to), m_inertial.noise);
        InertialState state = key_frames[i]->inertial();
        state.velocity = key_frames[i + 1]->inertial().velocity - alignment.gravity * summary.duration -
                         inertial_state(*key_frames[i]).rotation * summary.velocity;
        key_frames[i]->set_inertial(state);
    }

    m_metric_scale = alignment.scale;
    m_scale_uncertainty = alignment.scale_uncertainty;
    m_inertial.gravity = alignment.gravity;
    m_inertial.stream = &*m_imu;
    std::cout << "IMU alignment successful\n";
}

void Slam::initialize()
{
    auto result = initialize_map(
        m_video_loader, m_camera, m_static_mask, *m_feature_extractor, m_config, m_map, m_imu ? &*m_imu : nullptr);
    if (!result.ref_frame || !result.query_frame) {
        return;
    }

    m_frame_index = result.next_frame_index;
    m_mapper.adopt(result.ref_frame);
    m_mapper.adopt(result.query_frame);
    m_tracker.set_last_frame(result.query_frame);

    record_pose(*result.ref_frame);
    record_pose(*result.query_frame);
}

bool Slam::step()
{
    std::cout << "---------------------------------------- frame " << m_frame_index << "\n";
    m_diagnostics = FrameDiagnostics{};
    auto image = m_video_loader.get_next_frame();
    if (image.empty()) {
        std::cout << "No frame to process\n";
        return false;
    }

    // Recompute pose relative to the reference key frame
    if (m_tracker.has_last_frame() && m_tracker.last_frame().index() < m_trajectory.size()) {
        auto& last_frame = m_tracker.last_frame();
        last_frame.set_pose(m_trajectory.pose_at(last_frame.index()));
    }

    auto last_key_frame = m_mapper.key_frames().back();
    auto frame = m_tracker.track(image, m_frame_index++, m_trajectory, *last_key_frame, m_mapper.key_frames().size());

    std::shared_ptr<KeyFrame> new_key_frame;
    time_it("Create key frame", [&]() {
        if (m_mapper.needs_key_frame(*frame, m_tracker.tracks())) {
            std::cout << "Adding key frame after " << frame->index() - last_key_frame->index() << " frames\n";
            new_key_frame = m_mapper.insert(std::move(*frame), m_tracker.tracks(), m_trajectory, m_diagnostics);
            frame = new_key_frame;
        }
    });

    if (new_key_frame) {
        align_to_metric_scale();
        m_loop_detector.query(*new_key_frame, m_mapper.key_frames());
        if (m_loop_detector.consume_new_loop()) {
            std::cout << "----------LOOP CLOSED--------------\n";
            time_it("Pose graph opt", [&]() {
                m_diagnostics.loop_closed = optimization::pose_graph(m_mapper.key_frames(),
                                                                     m_loop_detector.constraints(),
                                                                     m_map,
                                                                     m_inertial.aligned(),
                                                                     m_inertial.gravity);
            });
            if (m_diagnostics.loop_closed) {
                auto& query = *m_mapper.key_frames().back();
                const auto& loops = m_loop_detector.constraints();
                const auto& constraint = loops.back();
                m_diagnostics.loop_correction =
                    (query.camera_center() - m_mapper.key_frames()[constraint.to]->camera_center()).norm();
                time_it("Fuse loop", [&]() { m_mapper.fuse_loop(query, constraint.inliers); });
                if (m_config.bundle_adjust) {
                    m_mapper.bundle_adjust(query);
                }
            }
        }
    }
    m_diagnostics.loop = m_loop_detector.last();
    m_diagnostics.loops = m_loop_detector.constraints().size();
    m_tracker.set_last_frame(frame);
    record_pose(*frame);
    m_diagnostics.map_size = m_map.size();

    m_tracker.tracks().extend(*frame, new_key_frame.get(), MAX_TRACK_SIGHTINGS);
    return true;
}

void Slam::record_pose(const Frame& frame)
{
    const auto& key_frames = m_mapper.key_frames();
    m_trajectory.record(frame, key_frames.empty() ? nullptr : key_frames.back().get());
}

float Slam::reprojection_error() const
{
    float error = 0.0;
    int num_projected = 0;
    for (const auto& frame : m_mapper.key_frames()) {
        for (const auto& match : frame->map_matches()) {
            const auto& point = match.point;
            auto projected = m_camera.project(frame->pose(), point.position());
            auto image_point =
                Eigen::Vector2f(frame->keypoint(match.keypoint_index).pt.x, frame->keypoint(match.keypoint_index).pt.y);
            error += (projected - image_point).stableNorm();
            num_projected++;
        }
    }
    return error / num_projected;
}

const FrameDiagnostics& Slam::diagnostics() const
{
    return m_diagnostics;
}

const Map& Slam::map() const
{
    return m_map;
}

const Frame& Slam::frame() const
{
    return m_tracker.last_frame();
}

std::vector<Eigen::Matrix4f> Slam::poses() const
{
    std::vector<Eigen::Matrix4f> poses;
    for (const auto& frame : m_mapper.key_frames()) {
        poses.push_back(frame->pose());
    }
    return poses;
}

std::vector<Eigen::Matrix4f> Slam::trajectory() const
{
    return m_trajectory.poses();
}

} // namespace slam
