#include "KeyFrame.h"

namespace slam {

KeyFrame::KeyFrame(size_t index, const Eigen::Matrix4f& pose, const ExtractedFeatures& features)
    : m_index(index), m_pose(pose), m_features(features)
{
}

size_t KeyFrame::index() const
{
    return m_index;
}

const Eigen::Matrix4f& KeyFrame::pose() const
{
    return m_pose;
}

const ExtractedFeatures& KeyFrame::features() const
{
    return m_features;
}

const cv::Mat KeyFrame::descriptor(size_t index) const
{
    return m_features.descriptors.row(index);
}

}