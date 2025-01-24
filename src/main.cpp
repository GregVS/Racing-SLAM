#include <iostream>
#include <yaml-cpp/yaml.h>

#include "Camera.h"
#include "Slam.h"
#include "VideoLoader.h"
#include "Visualization.h"

slam::Camera load_camera(const YAML::Node& config, slam::VideoLoader& video_loader)
{
    float fx = config["fx"].as<float>();
    float fy = config["fy"].as<float>();
    float cx = config["cx"] ? config["cx"].as<float>() : video_loader.get_width() / 2;
    float cy = config["cy"] ? config["cy"].as<float>() : video_loader.get_height() / 2;

    return slam::Camera(fx, fy, cx, cy, video_loader.get_width(), video_loader.get_height());
}

struct Setup {
    slam::Camera camera;
    cv::Mat mask;
    slam::VideoLoader video_loader;
};

Setup load_setup(const YAML::Node& config)
{
    std::string video_path = config["video"].as<std::string>();
    slam::VideoLoader video_loader(video_path);
    slam::Camera camera = load_camera(config, video_loader);

    cv::Mat mask = cv::Mat::ones(video_loader.get_height(), video_loader.get_width(), CV_8UC1);
    if (config["mask"]) {
        std::string mask_path = config["mask"].as<std::string>();
        mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
    }

    return {camera, mask, video_loader};
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <yaml_file>" << std::endl;
        return 1;
    }

    auto yaml_file = argv[1];
    YAML::Node yaml = YAML::LoadFile(yaml_file);
    Setup setup = load_setup(yaml);

    slam::SlamConfig config = {
        .triangulate_points = false,
        .bundle_adjust = true,
        .optimize_pose = false,
        .cull_points = false,
        .essential_matrix_estimation = false,
    };
    slam::Slam slam(setup.video_loader, setup.camera, setup.mask, config);
    slam.initialize();

    // Display the map matches
    slam::Visualization visualization("Racing SLAM");
    visualization.initialize();
    visualization.run_threaded();

    while (!visualization.has_quit()) {
        // Draw the camera poses
        auto poses = slam.poses();
        poses.push_back(slam.frame().pose());
        visualization.set_camera_poses(poses);

        // Draw the map points
        std::vector<slam::Visualization::Point> points;
        for (const auto& point : slam.map()) {
            points.push_back({point.position(), point.color()});
        }
        visualization.set_points(points);

        // Draw the keypoints and map matches on the frame
        const auto& frame = slam.frame();
        cv::Mat render = frame.image().clone();
        for (const auto& point : slam.map()) {
            auto uv = setup.camera.project(frame.pose(), point.position());
            cv::circle(render, cv::Point2f(uv[0], uv[1]), 1, cv::Scalar(0, 0, 255), -1);
        }
        std::vector<cv::KeyPoint> keypoints;
        for (const auto& match : frame.map_matches()) {
            auto feature = frame.keypoint(match.keypoint_index);
            keypoints.push_back(feature);
            auto uv = setup.camera.project(frame.pose(), match.point.position());
            cv::line(render, feature.pt, cv::Point2f(uv[0], uv[1]), cv::Scalar(0, 0, 0), 1);
        }
        cv::drawKeypoints(render,
                          keypoints,
                          render,
                          cv::Scalar(0, 255, 0),
                          cv::DrawMatchesFlags::DEFAULT);
        visualization.set_image(render);

        std::cout << "Reprojection error: " << slam.reprojection_error() << std::endl;

        // Next frame
        visualization.wait_for_keypress();
        slam.step();
    }

    return 0;
}
