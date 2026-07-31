#include "Runner.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <yaml-cpp/yaml.h>

#include "../Camera.h"
#include "../Frame.h"
#include "../Slam.h"
#include "../Trajectory.h"
#include "../Visualization.h"
#include "../features/OrbFeatureExtractor.h"

namespace slam::app {

namespace {

struct Setup {
    Camera camera;
    cv::Mat mask;
    VideoLoader video_loader;
};

Camera load_camera(const YAML::Node& config, VideoLoader& video_loader)
{
    auto fx = config["fx"].as<float>();
    auto fy = config["fy"].as<float>();
    auto cx = config["cx"] ? config["cx"].as<float>() : video_loader.get_width() / 2;
    auto cy = config["cy"] ? config["cy"].as<float>() : video_loader.get_height() / 2;

    return {fx, fy, cx, cy, video_loader.get_width(), video_loader.get_height()};
}

Setup load_setup(const YAML::Node& config)
{
    auto video_path = config["video"].as<std::string>();
    VideoLoader video_loader(video_path);
    Camera camera = load_camera(config, video_loader);

    cv::Mat mask = cv::Mat::ones(video_loader.get_height(), video_loader.get_width(), CV_8UC1);
    if (config["mask"]) {
        auto mask_path = config["mask"].as<std::string>();
        mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
    }

    return {camera, mask, video_loader};
}

void run_viewer(Slam& slam, const Camera& camera)
{
    std::atomic<bool> pause = false;

    Visualization visualization("Racing SLAM");
    visualization.initialize();
    visualization.set_pause_callback([&pause]() { pause = !pause; });
    visualization.run_threaded();

    while (!visualization.has_quit()) {
        // Draw the camera poses
        auto poses = slam.poses();
        poses.push_back(slam.frame().pose());
        visualization.set_camera_poses(poses);

        // Draw the map points
        std::vector<Visualization::Point> points;
        for (const auto& point : slam.map()) {
            points.push_back({point.position(), point.color()});
        }
        visualization.set_points(points);

        // Draw the keypoints and map matches on the frame
        const auto& frame = slam.frame();
        cv::Mat render = frame.image().clone();
        for (const auto& point : slam.map()) {
            auto uv = camera.project(frame.pose(), point.position());
            cv::circle(render, cv::Point2f(uv[0], uv[1]), 1, cv::Scalar(0, 0, 255), -1);
        }
        std::vector<cv::KeyPoint> keypoints;
        for (const auto& match : frame.map_matches()) {
            auto feature = frame.keypoint(match.keypoint_index);
            auto uv = camera.project(frame.pose(), match.point.position());
            if (!camera.is_in_image(uv)) {
                continue;
            }
            auto viewing_normal = match.point.avg_viewing_normal();
            auto viewing_direction = (match.point.position() - frame.camera_center()).normalized();
            auto dot = viewing_normal.dot(viewing_direction);
            if (dot < 0.5) {
                continue;
            }
            keypoints.push_back(feature);
            cv::line(render, feature.pt, cv::Point2f(uv[0], uv[1]), cv::Scalar(0, 0, 0), 1);
        }
        cv::drawKeypoints(
            render, keypoints, render, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
        visualization.set_image(render);

        std::cout << "Reprojection error: " << slam.reprojection_error() << std::endl;

        while (pause)
            ;
        slam.step();
    }
}

void write_point_cloud(const std::filesystem::path& path, const Slam& slam)
{
    size_t count = 0;
    for (const auto& point : slam.map()) {
        (void)point;
        count++;
    }
    std::ofstream ply(path);
    ply << "ply\nformat ascii 1.0\nelement vertex " << count << "\n"
        << "property float x\nproperty float y\nproperty float z\n"
        << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        << "property int frame\nend_header\n";
    for (const auto& point : slam.map()) {
        // The earliest observing key frame stands in for the creation frame
        size_t first = std::numeric_limits<size_t>::max();
        for (const auto& [frame, index] : point.observations()) {
            first = std::min(first, frame->index());
        }
        const auto& p = point.position();
        const auto& c = point.color();
        ply << p.x() << " " << p.y() << " " << p.z() << " " << int(c[0]) << " " << int(c[1])
            << " " << int(c[2]) << " " << first << "\n";
    }
}

bool write_outputs(const Options& options, const Slam& slam, int seed, double seconds_per_frame)
{
    std::filesystem::path run_dir = std::filesystem::path(options.output_dir) / options.run_id;
    std::filesystem::create_directories(run_dir);

    auto trajectory = slam.trajectory();
    auto trajectory_path = run_dir / (options.sequence + ".txt");
    if (!trajectory::write_kitti(trajectory_path.string(), trajectory)) {
        std::cerr << "Failed to write trajectory: " << trajectory_path << std::endl;
        return false;
    }

    std::ofstream meta(run_dir / (options.sequence + ".meta.yaml"));
    meta << "sequence: " << options.sequence << std::endl;
    meta << "run_id: " << options.run_id << std::endl;
    meta << "config: " << std::filesystem::absolute(options.config_path).string() << std::endl;
    meta << "seed: " << seed << std::endl;
    meta << "frames: " << trajectory.size() << std::endl;
    meta << "seconds_per_frame: " << seconds_per_frame << std::endl;
    meta << "reprojection_error: " << slam.reprojection_error() << std::endl;

    write_point_cloud(run_dir / (options.sequence + ".ply"), slam);

    std::cout << "Trajectory written to: " << trajectory_path.string() << std::endl;
    return true;
}

} // namespace

int run(const Options& options)
{
    YAML::Node yaml = YAML::LoadFile(options.config_path);

    // Fixed seed for reproducibility
    int seed = yaml["seed"] ? yaml["seed"].as<int>() : 0;
    cv::setRNGSeed(seed);
    std::cout << "Random seed: " << seed << std::endl;

    Setup setup = load_setup(yaml);

    SlamConfig config = {
        .triangulate_points = true,
        .bundle_adjust = true,
        .optimize_pose = true,
        .cull_points = true,
        .essential_matrix_estimation = true,
    };
    Slam slam(setup.video_loader,
              setup.camera,
              setup.mask,
              std::make_unique<features::OrbFeatureExtractor>(),
              config);
    slam.initialize();

    auto start = std::chrono::steady_clock::now();
    if (options.headless) {
        while (slam.step())
            ;
    } else {
        run_viewer(slam, setup.camera);
    }
    std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;

    std::cout << "Reprojection error: " << slam.reprojection_error() << std::endl;

    if (!options.output_dir.empty()) {
        size_t frames = slam.trajectory().size();
        double seconds_per_frame = frames > 0 ? elapsed.count() / frames : 0.0;
        if (!write_outputs(options, slam, seed, seconds_per_frame)) {
            return 1;
        }
    }
    return 0;
}

} // namespace slam::app
