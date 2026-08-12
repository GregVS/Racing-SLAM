#include "ImuStream.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace slam::imu {

namespace {

/** EuRoC IMU data format */
bool parse_row(const std::string& line, Sample& sample)
{
    if (line.empty() || line[0] == '#') {
        return false;
    }

    std::string field;
    std::istringstream row(line);
    double values[7];
    for (double& value : values) {
        if (!std::getline(row, field, ',')) {
            return false;
        }
        value = std::stod(field);
    }

    sample.time = values[0] * 1e-9;
    sample.gyro = Eigen::Vector3d(values[1], values[2], values[3]);
    sample.accel = Eigen::Vector3d(values[4], values[5], values[6]);
    return true;
}

Sample interpolate(const Sample& before, const Sample& after, double time)
{
    const double span = after.time - before.time;
    const double fraction = span > 0.0 ? (time - before.time) / span : 0.0;
    return {time,
            before.gyro + fraction * (after.gyro - before.gyro),
            before.accel + fraction * (after.accel - before.accel)};
}

double scalar(const YAML::Node& node, const std::string& key, double fallback)
{
    return node[key] ? node[key].as<double>() : fallback;
}

Eigen::Matrix4d transform(const YAML::Node& node)
{
    if (!node || !node["data"] || node["data"].size() != 16) {
        return Eigen::Matrix4d::Identity();
    }
    Eigen::Matrix4d matrix;
    for (int i = 0; i < 16; i++) {
        matrix(i / 4, i % 4) = node["data"][i].as<double>();
    }
    return matrix;
}

} // namespace

Stream Stream::load(const std::string& directory)
{
    const std::filesystem::path root(directory);

    std::ifstream csv(root / "data.csv");
    if (!csv) {
        throw std::runtime_error("no imu data at " + (root / "data.csv").string());
    }

    Stream stream;
    std::string line;
    Sample sample;
    while (std::getline(csv, line)) {
        if (parse_row(line, sample)) {
            stream.m_samples.push_back(sample);
        }
    }
    if (stream.m_samples.size() < 2) {
        throw std::runtime_error("imu stream at " + directory + " holds less than two samples");
    }

    // Load sensor specs
    const std::filesystem::path sidecar = root / "sensor.yaml";
    if (std::filesystem::exists(sidecar)) {
        const YAML::Node config = YAML::LoadFile(sidecar.string());
        Calibration& calibration = stream.m_calibration;
        calibration.rate = scalar(config, "rate_hz", 0.0);
        calibration.noise.gyro = scalar(config, "gyroscope_noise_density", 0.0);
        calibration.noise.accel = scalar(config, "accelerometer_noise_density", 0.0);
        calibration.noise.gyro_bias = scalar(config, "gyroscope_random_walk", 0.0);
        calibration.noise.accel_bias = scalar(config, "accelerometer_random_walk", 0.0);
        calibration.sensor_to_camera = transform(config["T_SC"]);
    }
    return stream;
}

std::vector<Sample> Stream::between(double start, double end) const
{
    std::vector<Sample> slice;
    start = std::max(start, first());
    end = std::min(end, last());
    if (m_samples.size() < 2 || end <= start) {
        return slice;
    }

    const auto at = [this](double time) {
        const auto after = std::upper_bound(
            m_samples.begin(), m_samples.end(), time, [](double t, const Sample& s) { return t < s.time; });
        // upper_bound only reaches end() when time is the last sample exactly since interval is clipped to the data
        const auto before = after == m_samples.begin() ? after : std::prev(after);
        return after == m_samples.end() ? m_samples.back() : interpolate(*before, *after, time);
    };

    slice.push_back(at(start));
    const auto tail = std::upper_bound(
        m_samples.begin(), m_samples.end(), start, [](double t, const Sample& s) { return t < s.time; });
    for (auto it = tail; it != m_samples.end() && it->time < end; ++it) {
        slice.push_back(*it);
    }
    slice.push_back(at(end));
    return slice;
}

} // namespace slam::imu
