#pragma once

#include <Eigen/Dense>
#include <vector>

namespace slam {

class Frame;

class Trajectory {
  public:
    void record(const Frame& frame, const Frame* reference);

    Eigen::Matrix4f pose_at(size_t index) const;
    size_t size() const;
    std::vector<Eigen::Matrix4f> poses() const;

  private:
    struct Entry {
        const Frame* reference = nullptr; // null before the first key frame exists
        Eigen::Matrix4f relative = Eigen::Matrix4f::Identity();
    };

    std::vector<Entry> m_entries;
};

} // namespace slam
