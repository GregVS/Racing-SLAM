#pragma once

#include "Camera.h"
#include "FeatureExtractor.h"
#include "Frame.h"
#include "PoseEstimator.h"
#include "VideoLoader.h"

struct TestData {
    slam::VideoLoader video_loader;
    slam::FeatureExtractor feature_extractor;
    slam::PoseEstimator pose_estimator;
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

static TestData load_test_data()
{
    TestData test_data = {
        .video_loader = slam::VideoLoader("videos/lime-rock-race.mp4"),
        .feature_extractor = slam::FeatureExtractor(),
        .pose_estimator = slam::PoseEstimator(),
        .camera = slam::Camera(914,
                               test_data.video_loader.get_width(),
                               test_data.video_loader.get_height()),
    };
    cv::Mat mask = cv::Mat::zeros(test_data.video_loader.get_height(),
                                  test_data.video_loader.get_width(),
                                  CV_8UC1);
    mask(cv::Rect(0,
                  0,
                  test_data.video_loader.get_width(),
                  test_data.video_loader.get_height() * 0.7)) = 255;
    test_data.static_mask = mask;
    return test_data;
}