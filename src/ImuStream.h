#pragma once

#include <string>
#include <vector>

#include "Imu.h"

namespace slam::imu {

class Stream {
  public:
    /** Reads EuRoC-format data.csv */
    static Stream load(const std::string& csv_path);

    /** Interval is [start, end] */
    std::vector<Sample> between(double start, double end) const;

    const std::vector<Sample>& samples() const
    {
        return m_samples;
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
};

} // namespace slam::imu
