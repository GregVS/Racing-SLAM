#include "ImuStream.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

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

} // namespace

Stream Stream::load(const std::string& csv_path)
{
    std::ifstream csv(csv_path);
    if (!csv) {
        throw std::runtime_error("no imu data at " + csv_path);
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
        throw std::runtime_error("imu stream at " + csv_path + " holds less than two samples");
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
