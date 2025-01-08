#include "Visualization.h"

#include <pangolin/pangolin.h>

namespace slam {

Visualization::Visualization(const std::string& window_name)
    : m_window_name(window_name), m_camera_state(nullptr), m_handler(nullptr), m_display(nullptr),
      m_initialized(false)
{
}

Visualization::~Visualization()
{
    if (m_camera_state)
        delete m_camera_state;
    if (m_handler)
        delete m_handler;
    // m_display should be deleted by Pangolin
}

void Visualization::initialize(int width, int height)
{
    pangolin::CreateWindowAndBind(m_window_name, width, height);
    glEnable(GL_DEPTH_TEST);

    m_camera_state = new pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(width, height, 420, 420, width / 2, height / 2, 0.2, 1000),
        pangolin::ModelViewLookAt(-2, -2, -2, 0, 0, 5, pangolin::AxisNegY));

    m_handler = new pangolin::Handler3D(*m_camera_state, pangolin::AxisNegY);
    m_display = &pangolin::CreateDisplay().SetBounds(0.0, 1.0, 0.0, 1.0).SetHandler(m_handler);

    m_initialized = true;
}

void Visualization::draw_camera_pose(const Eigen::Matrix4f& pose)
{
    const float camera_size = 0.2f;
    const Eigen::Matrix4f inverse_pose = pose.inverse();

    glPushMatrix();
    glMultMatrixf(inverse_pose.data());

    glColor3f(0.0f, 0.0f, 1.0f);
    glBegin(GL_TRIANGLES);
    glVertex3f(0, 0, 0);
    glVertex3f(camera_size, 0, -camera_size);
    glVertex3f(-camera_size, 0, -camera_size);
    glEnd();

    glPopMatrix();
}

void Visualization::set_camera_poses(const std::vector<Eigen::Matrix4f>& poses)
{
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_poses = poses;
}

void Visualization::set_points(const std::vector<Eigen::Vector3f>& points)
{
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_points = points;
}

void Visualization::draw_points(const std::vector<Eigen::Vector3f>& points)
{
    glPointSize(5);
    glBegin(GL_POINTS);
    for (const auto& point : points) {
        glVertex3f(point[0], point[1], point[2]);
    }
    glEnd();
}

void Visualization::run()
{
    initialize();
    while (!pangolin::ShouldQuit()) {
        m_display->Activate(*m_camera_state);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw coordinate frame
        glLineWidth(3);
        pangolin::glDrawAxis(3.0);

        {
            std::lock_guard<std::mutex> lock(m_data_mutex);
            for (const auto& pose : m_poses) {
                draw_camera_pose(pose);
            }
            draw_points(m_points);
        }

        pangolin::FinishFrame();
    }
}

} // namespace slam