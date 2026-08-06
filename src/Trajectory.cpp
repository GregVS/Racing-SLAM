#include "Trajectory.h"

#include "Frame.h"

namespace slam {

void Trajectory::record(const Frame& frame, const Frame* reference)
{
    // Frames dropped during initialization keep the previous pose, so indexing stays by frame index
    auto fill = m_entries.empty() ? Entry{} : m_entries.back();
    m_entries.resize(frame.index() + 1, fill);

    // World to camera, so relative to the reference is Tcr = Tcw * Twr
    m_entries[frame.index()] = {reference, reference ? frame.pose() * reference->pose().inverse() : frame.pose()};
}

Eigen::Matrix4f Trajectory::pose_at(size_t index) const
{
    const auto& entry = m_entries[index];
    return entry.reference ? entry.relative * entry.reference->pose() : entry.relative;
}

size_t Trajectory::size() const
{
    return m_entries.size();
}

std::vector<Eigen::Matrix4f> Trajectory::poses() const
{
    std::vector<Eigen::Matrix4f> poses;
    poses.reserve(m_entries.size());
    for (size_t i = 0; i < m_entries.size(); i++) {
        poses.push_back(pose_at(i));
    }
    return poses;
}

} // namespace slam
