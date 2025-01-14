#pragma once

#include "Camera.h"
#include "Features.h"
#include "Frame.h"
#include "PoseEstimation.h"
#include "VideoLoader.h"

struct TestData {
    slam::VideoLoader video_loader;
    slam::Camera camera;
    cv::Mat static_mask;
};

template <typename T> T time_it(const std::string& name, std::function<T()> func)
{
    auto start = std::chrono::high_resolution_clock::now();
    T result = func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << name << " took " << duration.count() * 1000 << " ms" << std::endl;
    return result;
}

constexpr std::string_view LIME_ROCK_RACE_VIDEO = "videos/lime-rock-race.mp4";
constexpr std::string_view HIGHWAY_VIDEO = "videos/highway-lowres.mp4";
constexpr std::string_view OKAYAMA_VIDEO = "videos/okayama-lap.mp4";

static TestData load_test_data(std::string_view video = LIME_ROCK_RACE_VIDEO)
{
    auto video_loader = slam::VideoLoader(std::string(video));

    float FOV;
    if (video == HIGHWAY_VIDEO) {
        FOV = 120;
    } else if (video == LIME_ROCK_RACE_VIDEO) {
        FOV = 70;
    } else if (video == OKAYAMA_VIDEO) {
        FOV = 120;
    }

    int focal_length = (float) video_loader.get_width() / (2.0 * tan(FOV * M_PI / 360));
    std::cout << "Focal length: " << focal_length << std::endl;
    TestData test_data = {
        .video_loader = video_loader,
        .camera = slam::Camera(focal_length,
                               test_data.video_loader.get_width(),
                               test_data.video_loader.get_height())
    };

    cv::Mat mask = cv::Mat::zeros(test_data.video_loader.get_height(),
                                  test_data.video_loader.get_width(),
                                  CV_8UC1);
    if (video == LIME_ROCK_RACE_VIDEO) {
        mask(cv::Rect(0, 0, mask.cols, mask.rows * 0.7)) = 255;
    } else if (video == HIGHWAY_VIDEO) {
        mask(cv::Rect(0, 0, mask.cols, mask.rows)) = 255;
    } else if (video == OKAYAMA_VIDEO) {
        mask = cv::imread("videos/okayama-lap-mask.png", cv::IMREAD_GRAYSCALE);
    }
    test_data.static_mask = mask;

    return test_data;
}