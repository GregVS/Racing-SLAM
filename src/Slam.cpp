#include "Slam.h"

#include <fstream>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

#include "features/FeatureExtractor.h"
#include "Helpers.h"
#include "Init.h"
#include "Optimization.h"
#include "PoseEstimation.h"
#include "Triangulation.h"

namespace slam {

// Number of most recent key frames whose poses local bundle adjustment is allowed to move
static const size_t BA_WINDOW = 10;

// Key frame spacing
static const size_t MIN_KEY_FRAME_GAP = 5;
static const size_t MAX_KEY_FRAME_GAP = 20;
static const size_t MIN_TRACKED_POINTS = 50;

Slam::Slam(const VideoLoader& video_loader,
           const Camera& camera,
           const cv::Mat& image_mask,
           std::unique_ptr<features::BaseFeatureExtractor> feature_extractor,
           const SlamConfig& config)
    : m_video_loader(video_loader), m_camera(camera), m_static_mask(image_mask),
      m_feature_extractor(std::move(feature_extractor)), m_config(config)
{
}

std::optional<Frame> Slam::process_next_frame()
{
    auto image = m_video_loader.get_next_frame();
    if (image.empty()) {
        return std::nullopt;
    }
    auto features = m_feature_extractor->extract_features(image, m_static_mask);
    return std::make_optional<Frame>(m_frame_index++, image, features);
}

void Slam::initialize()
{
    // First first frames
    auto maybe_result = init::find_initializing_frames([this]() { return process_next_frame(); },
                                                       m_camera,
                                                       *m_feature_extractor);
    if (!maybe_result) {
        std::cout << "Initialization failed" << std::endl;
        return;
    }

    auto result = std::move(*maybe_result);
    auto ref_frame = std::make_shared<Frame>(std::move(result.ref_frame));
    auto query_frame = std::make_shared<Frame>(std::move(result.query_frame));
    std::cout << "Initializing frames: " << ref_frame->index() << " and " << query_frame->index()
              << std::endl;

    // Triangulate points
    auto matches = m_feature_extractor->match_features(ref_frame->features(),
                                                       query_frame->features());
    std::cout << "Number of matches: " << matches.size() << std::endl;
    auto points = triangulation::triangulate_points(*ref_frame, *query_frame, matches, m_camera);

    // Add points to map
    for (int i = 0; i < points.size(); i++) {
        auto match = matches[points[i].match_index];
        m_map.create_point(points[i].position, *ref_frame, *query_frame, match);
    }
    std::cout << "Number of triangulated points: " << points.size() << std::endl;

    // Bundle adjustment
    {
        auto config = optimization::OptimizationConfig{
            .optimize_points = true,
            .frames = {{false, ref_frame.get()}, {true, query_frame.get()}},
        };
        optimization::optimize(config, m_camera, m_map);

        float scale = 1.0f /
                      (query_frame->pose().block<3, 1>(0, 3) - ref_frame->pose().block<3, 1>(0, 3))
                          .stableNorm();
        std::cout << "Scale: " << scale << std::endl;

        auto scaled_pose = query_frame->pose();
        scaled_pose.block<3, 1>(0, 3) = query_frame->pose().block<3, 1>(0, 3) * scale;
        query_frame->set_pose(scaled_pose);
        for (auto& point : m_map) {
            point.set_position(point.position() * scale);
        }
    }

    // Add frames to key frames
    m_key_frames.push_back(ref_frame);
    m_key_frames.push_back(query_frame);
    m_last_frame = query_frame;

    record_pose(*ref_frame);
    record_pose(*query_frame);
}

bool Slam::step()
{
    std::cout << "----------------------------------------" << std::endl;
    auto maybe_frame = process_next_frame();
    if (!maybe_frame) {
        std::cout << "No frame to process" << std::endl;
        return false;
    }

    auto frame = std::make_shared<Frame>(std::move(*maybe_frame));
    auto last_key_frame = m_key_frames.back();

    // Initial pose estimation
    time_it("Initial pose estimation", [&]() { initial_pose_estimate(*frame); });

    // Match with map from last key frame
    time_it("Match with last key frame", [&]() { match_with_last_key_frame(*frame); });
    time_it("Optimize pose", [&]() { optimize_pose(*frame); });

    // Match with all map points
    time_it("Match with map", [&]() { match_with_map(*frame); });
    time_it("Optimize pose", [&]() { optimize_pose(*frame); });

    // Create key frame if needed
    time_it("Create key frame", [&]() {
        if (needs_key_frame(*frame, *last_key_frame)) {
            std::cout << "Adding key frame after " << frame->index() - last_key_frame->index()
                      << " frames" << std::endl;
            init_key_frame(*frame);
            m_key_frames.push_back(frame);
        }
    });

    m_last_frame = frame;
    record_pose(*frame);
    return true;
}

bool Slam::needs_key_frame(const Frame& frame, const Frame& last_key_frame) const
{
    if (frame.num_map_matches() < MIN_TRACKED_POINTS) {
        return true;
    }

    size_t gap = frame.index() - last_key_frame.index();
    if (gap < MIN_KEY_FRAME_GAP) {
        return false;
    }

    return gap >= MAX_KEY_FRAME_GAP || frame.num_map_matches() < 0.9 * last_key_frame.num_map_matches();
}

void Slam::record_pose(const Frame& frame)
{
    // Frames dropped during initialization keep the previous pose so that the trajectory stays
    // indexed by frame index
    auto fill = m_trajectory.empty() ? Eigen::Matrix4f::Identity() : m_trajectory.back();
    m_trajectory.resize(frame.index() + 1, fill);
    m_trajectory[frame.index()] = frame.pose();
}

void Slam::initial_pose_estimate(Frame& frame)
{
    if (m_config.essential_matrix_estimation || m_key_frames.size() < 2) {
        auto pose_estimate = pose::estimate_pose(
            m_last_frame->features(),
            frame.features(),
            m_feature_extractor->match_features(m_last_frame->features(), frame.features()),
            m_camera);
        frame.set_pose(pose_estimate.pose * m_last_frame->pose());
    } else {
        frame.set_pose(m_last_frame->pose());
    }
}

void Slam::match_with_last_key_frame(Frame& frame)
{
    auto last_frame = m_key_frames.back();
    auto map_matches = m_feature_extractor->match_features(
        frame,
        m_camera,
        m_map,
        [&](const MapPoint& point) { return point.is_observed_by(last_frame.get()); });
    for (const auto& match : map_matches) {
        frame.add_map_match(match);
    }
    std::cout << "Map matches with last frame: " << map_matches.size() << std::endl;
}

void Slam::match_with_map(Frame& frame)
{
    auto map_matches = m_feature_extractor->match_features(
        frame,
        m_camera,
        m_map,
        [&](const MapPoint& point) { return true; });
    for (const auto& match : map_matches) {
        frame.add_map_match(match);
    }
    std::cout << "Number of map matches: " << map_matches.size() << std::endl;
}

void Slam::optimize_pose(Frame& frame)
{
    if (!m_config.optimize_pose) {
        return;
    }
    // Motion-only BA
    auto config = optimization::OptimizationConfig{
        .optimize_points = false,
        .frames = {{true, &frame}},
    };
    optimization::optimize(config, m_camera, m_map);
}

void Slam::init_key_frame(Frame& frame)
{
    // Add map associations
    auto last_key_frame = m_key_frames.back();
    for (const auto& match : frame.map_matches()) {
        m_map.add_association(frame, match);
    }

    // Triangulate unmatched points
    if (m_config.triangulate_points) {
        auto feature_matches = m_feature_extractor->match_features(last_key_frame->features(),
                                                                   frame.features());
        auto unmatched = features::unmatched_features(*last_key_frame, frame, feature_matches);
        auto points = triangulation::triangulate_points(*last_key_frame,
                                                        frame,
                                                        unmatched,
                                                        m_camera);
        for (int i = 0; i < points.size(); i++) {
            auto match = unmatched[points[i].match_index];
            m_map.create_point(points[i].position, *last_key_frame, frame, match);
        }
        std::cout << "Number of triangulated points: " << points.size() << std::endl;
    }

    // Bundle adjustment
    if (m_config.bundle_adjust) {
        size_t first_optimized = m_key_frames.size() > BA_WINDOW ? m_key_frames.size() - BA_WINDOW
                                                                 : 2;
        std::vector<optimization::FrameConfig> frame_configs;
        for (size_t i = 0; i < m_key_frames.size(); i++) {
            frame_configs.push_back({i >= first_optimized, m_key_frames[i].get()});
        }
        frame_configs.push_back({true, &frame});
        auto config = optimization::OptimizationConfig{
            .optimize_points = true,
            .frames = frame_configs,
        };
        optimization::optimize(config, m_camera, m_map);
    }

    // Cull points
    if (m_config.cull_points) {
        cull_points();
    }
}

void Slam::cull_points()
{
    std::vector<MapPoint*> points_to_remove;
    for (auto& point : m_map) {
        float error = 0.0;
        int num_projected = 0;
        for (const auto& [frame, index] : point.observations()) {
            auto projected = m_camera.project(frame->pose(), point.position());
            auto image_point = Eigen::Vector2f(frame->keypoint(index).pt.x,
                                               frame->keypoint(index).pt.y);
            error += (projected - image_point).norm();
            num_projected++;
        }
        if (error / num_projected > 3.0) {
            points_to_remove.push_back(&point);
        }
    }

    std::cout << "Number of points to remove: " << points_to_remove.size() << std::endl;
    for (const auto& point : points_to_remove) {
        m_map.remove_point(point);
    }
}

float Slam::reprojection_error() const
{
    float error = 0.0;
    int num_projected = 0;
    for (const auto& frame : m_key_frames) {
        for (const auto& match : frame->map_matches()) {
            auto point = match.point;
            auto projected = m_camera.project(frame->pose(), point.position());
            auto image_point = Eigen::Vector2f(frame->keypoint(match.keypoint_index).pt.x,
                                               frame->keypoint(match.keypoint_index).pt.y);
            error += (projected - image_point).stableNorm();
            num_projected++;
        }
    }
    return error / num_projected;
}

const Map& Slam::map() const
{
    return m_map;
}

const Frame& Slam::frame() const
{
    return *m_last_frame;
}

std::vector<Eigen::Matrix4f> Slam::poses() const
{
    std::vector<Eigen::Matrix4f> poses;
    for (const auto& frame : m_key_frames) {
        poses.push_back(frame->pose());
    }
    return poses;
}

std::vector<Eigen::Matrix4f> Slam::trajectory() const
{
    // Key frames are refined by bundle adjustment after they were recorded
    auto trajectory = m_trajectory;
    for (const auto& frame : m_key_frames) {
        if (frame->index() < trajectory.size()) {
            trajectory[frame->index()] = frame->pose();
        }
    }
    return trajectory;
}

void Slam::save_state(const std::string& filename) const
{
    using json = nlohmann::json;

    // Create mapping from MapPoint pointers to sequential IDs
    std::unordered_map<const MapPoint*, size_t> map_point_to_id;
    size_t map_point_id = 0;
    for (const auto& point : m_map) {
        map_point_to_id[&point] = map_point_id++;
    }

    // Create mapping from key frame pointers to indices
    std::unordered_map<const Frame*, size_t> frame_to_index;
    for (size_t i = 0; i < m_key_frames.size(); i++) {
        frame_to_index[m_key_frames[i].get()] = i;
    }

    // Serialize key frames
    json key_frames_json = json::array();
    for (size_t frame_idx = 0; frame_idx < m_key_frames.size(); frame_idx++) {
        const auto& frame = m_key_frames[frame_idx];
        const auto& features = frame->features();

        // Serialize pose (4x4 matrix as 16-element array)
        json pose_json = json::array();
        const auto& pose = frame->pose();
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                pose_json.push_back(pose(i, j));
            }
        }

        // Serialize keypoints
        json keypoints_json = json::array();
        for (const auto& kp : features.keypoints) {
            json kp_json = {
                {"x", kp.pt.x},
                {"y", kp.pt.y},
                {"size", kp.size},
                {"angle", kp.angle},
                {"response", kp.response},
                {"octave", kp.octave},
                {"class_id", kp.class_id}
            };
            keypoints_json.push_back(kp_json);
        }

        // Serialize descriptors (cv::Mat as 2D array)
        json descriptors_json = json::array();
        if (!features.descriptors.empty()) {
            for (int i = 0; i < features.descriptors.rows; i++) {
                json row = json::array();
                for (int j = 0; j < features.descriptors.cols; j++) {
                    if (features.descriptors.type() == CV_8U) {
                        row.push_back(features.descriptors.at<uchar>(i, j));
                    } else if (features.descriptors.type() == CV_32F) {
                        row.push_back(features.descriptors.at<float>(i, j));
                    } else {
                        row.push_back(0); // Fallback
                    }
                }
                descriptors_json.push_back(row);
            }
        }

        // Serialize map matches (keypoint_index -> map_point_id)
        json map_matches_json = json::object();
        for (const auto& match : frame->map_matches()) {
            auto it = map_point_to_id.find(&match.point);
            if (it != map_point_to_id.end()) {
                map_matches_json[std::to_string(match.keypoint_index)] = it->second;
            }
        }

        json frame_json = {
            {"index", frame->index()},
            {"pose", pose_json},
            {"keypoints", keypoints_json},
            {"descriptors", descriptors_json},
            {"map_matches", map_matches_json}
        };
        key_frames_json.push_back(frame_json);
    }

    // Serialize map points
    json map_points_json = json::array();
    for (const auto& point : m_map) {
        // Serialize position (3D vector)
        json position_json = {point.position().x(), point.position().y(), point.position().z()};

        // Serialize color (RGB)
        const auto& color = point.color();
        json color_json = {color[0], color[1], color[2]};

        // Serialize observations (key_frame_index -> keypoint_index)
        json observations_json = json::object();
        for (const auto& [frame_ptr, keypoint_idx] : point.observations()) {
            auto it = frame_to_index.find(frame_ptr);
            if (it != frame_to_index.end()) {
                observations_json[std::to_string(it->second)] = keypoint_idx;
            }
        }

        json point_json = {
            {"position", position_json},
            {"color", color_json},
            {"observations", observations_json}
        };
        map_points_json.push_back(point_json);
    }

    // Create final JSON structure
    json state_json = {
        {"frame_index", m_frame_index},
        {"key_frames", key_frames_json},
        {"map_points", map_points_json}
    };

    // Write to file
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    file << state_json.dump();
    file.close();
    std::cout << "SLAM state saved to: " << filename << std::endl;
}

bool Slam::load_state(const std::string& filename)
{
    using json = nlohmann::json;

    // Read JSON file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return false;
    }

    json state_json;
    try {
        file >> state_json;
    } catch (const json::parse_error& e) {
        std::cerr << "Failed to parse JSON: " << e.what() << std::endl;
        return false;
    }
    file.close();

    // Clear current state
    m_map = Map();
    m_key_frames.clear();
    m_last_frame = nullptr;

    // Load frame index
    if (!state_json.contains("frame_index")) {
        std::cerr << "Missing frame_index in JSON" << std::endl;
        return false;
    }
    m_frame_index = state_json["frame_index"].get<size_t>();

    // Load map points first and create ID->MapPoint* mapping
    if (!state_json.contains("map_points")) {
        std::cerr << "Missing map_points in JSON" << std::endl;
        return false;
    }

    // First, create all map points and store them temporarily
    std::vector<std::unique_ptr<MapPoint>> temp_map_points;
    for (const auto& point_json : state_json["map_points"]) {
        // Load position
        if (!point_json.contains("position") || point_json["position"].size() != 3) {
            std::cerr << "Invalid position in map point" << std::endl;
            return false;
        }
        Eigen::Vector3f position(
            point_json["position"][0].get<float>(),
            point_json["position"][1].get<float>(),
            point_json["position"][2].get<float>()
        );

        // Create map point
        auto map_point = std::make_unique<MapPoint>(position);

        // Load color
        if (point_json.contains("color") && point_json["color"].size() == 3) {
            cv::Vec3b color(
                point_json["color"][0].get<uchar>(),
                point_json["color"][1].get<uchar>(),
                point_json["color"][2].get<uchar>()
            );
            map_point->set_color(color);
        }

        temp_map_points.push_back(std::move(map_point));
    }

    // Now add all points to map and create ID mapping
    std::vector<MapPoint*> map_points_by_id;
    for (auto& point : temp_map_points) {
        MapPoint* point_ptr = point.get();
        m_map.add_point(std::move(point));
        map_points_by_id.push_back(point_ptr);
    }

    // Load key frames
    if (!state_json.contains("key_frames")) {
        std::cerr << "Missing key_frames in JSON" << std::endl;
        return false;
    }

    std::vector<std::shared_ptr<Frame>> loaded_frames;
    for (const auto& frame_json : state_json["key_frames"]) {
        if (!frame_json.contains("index")) {
            std::cerr << "Missing index in key frame" << std::endl;
            return false;
        }
        size_t frame_index = frame_json["index"].get<size_t>();

        // Get image from video
        cv::Mat image = m_video_loader.get_frame(frame_index);
        if (image.empty()) {
            std::cerr << "Failed to load frame " << frame_index << " from video" << std::endl;
            return false;
        }

        // Reconstruct keypoints
        if (!frame_json.contains("keypoints")) {
            std::cerr << "Missing keypoints in key frame" << std::endl;
            return false;
        }
        std::vector<cv::KeyPoint> keypoints;
        for (const auto& kp_json : frame_json["keypoints"]) {
            cv::KeyPoint kp;
            kp.pt.x = kp_json["x"].get<float>();
            kp.pt.y = kp_json["y"].get<float>();
            kp.size = kp_json.contains("size") ? kp_json["size"].get<float>() : 0.0f;
            kp.angle = kp_json.contains("angle") ? kp_json["angle"].get<float>() : -1.0f;
            kp.response = kp_json.contains("response") ? kp_json["response"].get<float>() : 0.0f;
            kp.octave = kp_json.contains("octave") ? kp_json["octave"].get<int>() : 0;
            kp.class_id = kp_json.contains("class_id") ? kp_json["class_id"].get<int>() : -1;
            keypoints.push_back(kp);
        }

        // Reconstruct descriptors
        cv::Mat descriptors;
        if (frame_json.contains("descriptors") && !frame_json["descriptors"].empty()) {
            const auto& desc_json = frame_json["descriptors"];
            int rows = desc_json.size();
            if (rows > 0) {
                int cols = desc_json[0].size();
                // Try to determine type from first value (assuming consistent type)
                // Default to CV_8U for ORB descriptors
                descriptors = cv::Mat(rows, cols, CV_8U);
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        descriptors.at<uchar>(i, j) = desc_json[i][j].get<uchar>();
                    }
                }
            }
        }

        // Create ExtractedFeatures
        ExtractedFeatures features;
        features.keypoints = keypoints;
        features.descriptors = descriptors;

        // Create frame
        auto frame = std::make_shared<Frame>(frame_index, image, features);

        // Load pose
        if (frame_json.contains("pose") && frame_json["pose"].size() == 16) {
            Eigen::Matrix4f pose;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    pose(i, j) = frame_json["pose"][i * 4 + j].get<float>();
                }
            }
            frame->set_pose(pose);
        }

        // Load map matches and rebuild associations
        if (frame_json.contains("map_matches")) {
            for (const auto& [keypoint_str, map_point_id] : frame_json["map_matches"].items()) {
                size_t keypoint_index = std::stoul(keypoint_str);
                size_t point_id = map_point_id.get<size_t>();
                
                if (point_id >= map_points_by_id.size()) {
                    std::cerr << "Invalid map point ID: " << point_id << std::endl;
                    continue;
                }
                
                MapPoint* point_ptr = map_points_by_id[point_id];
                MapPointMatch match{*point_ptr, keypoint_index};
                m_map.add_association(*frame, match);
            }
        }

        loaded_frames.push_back(frame);
    }

    // Observations are already set up by add_association above, so we don't need to rebuild them separately

    // Set key frames and last frame
    m_key_frames = loaded_frames;
    if (!m_key_frames.empty()) {
        m_last_frame = m_key_frames.back();
    }

    std::cout << "SLAM state loaded from: " << filename << std::endl;
    std::cout << "  Frame index: " << m_frame_index << std::endl;
    std::cout << "  Key frames: " << m_key_frames.size() << std::endl;
    std::cout << "  Map points: " << map_points_by_id.size() << std::endl;

    return true;
}

} // namespace slam