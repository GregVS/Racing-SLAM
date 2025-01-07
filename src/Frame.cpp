#include "Frame.h"

namespace slam {

Frame::Frame(int index, const cv::Mat& image) : m_index(index), m_image(image) {}

int Frame::index() const { return m_index; }
const cv::Mat& Frame::image() const { return m_image; }

ExtractedFeatures Frame::set_features(const ExtractedFeatures& features)
{
    m_features = features;
    return m_features;
}

const ExtractedFeatures& Frame::features() const { return m_features; }

} // namespace slam