#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "Imu.h"

namespace slam::imu {

struct Calibration {
    double rate = 0.0; // sample rate in Hz
    NoiseDensity noise{0.0, 0.0, 0.0, 0.0};

    // Not needed for iRacing, but might be useful for EuRoC
    Eigen::Matrix4d sensor_to_camera = Eigen::Matrix4d::Identity();
};

class Stream {
  public:
    /** Assumes <directory>/data.csv and <directory>/sensor.yaml exist */
    static Stream load(const std::string& directory);

    /** Interval is [start, end] */
    std::vector<Sample> between(double start, double end) const;

    const std::vector<Sample>& samples() const
    {
        return m_samples;
    }
    const Calibration& calibration() const
    {
        return m_calibration;
    }
    double first() const
    {
        return m_samples.empty() ? 0.0 : m_samples.front().time;
    }
    double last() const
    {
        return m_samples.empty() ? 0.0 : m_samples.back().time;
    }
    size_t size() const
    {
        return m_samples.size();
    }

  private:
    std::vector<Sample> m_samples;
    Calibration m_calibration;
};

} // namespace slam::imu
