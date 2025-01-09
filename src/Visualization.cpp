#include "Visualization.h"

#include <chrono>
#include <pangolin/pangolin.h>
#include <thread>

namespace slam {

Visualization::Visualization(const std::string& window_name) : m_window_name(window_name) {}

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
    m_3d_display = &pangolin::CreateDisplay().SetBounds(0.4, 1.0, 0.0, 1.0).SetHandler(m_handler);
    m_image_display = &pangolin::CreateDisplay().SetBounds(0.0, 0.4, 0.0, 1.0);

    pangolin::GetBoundWindow()->RemoveCurrent();
    m_initialized = true;
    m_has_quit = false;
}

void Visualization::draw_camera_pose(const Eigen::Matrix4f& pose)
{
    const float camera_size = 1.0f;
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

void Visualization::wait_for_keypress()
{
    std::unique_lock<std::mutex> lock(m_key_pressed_mutex);
    m_key_pressed_cv.wait(lock);
}

void Visualization::set_image(const cv::Mat& image)
{
    std::lock_guard<std::mutex> lock(m_render_lock);
    cv::cvtColor(image, m_image, cv::COLOR_BGR2RGB);
    m_image_texture = nullptr;
}

void Visualization::set_camera_poses(const std::vector<Eigen::Matrix4f>& poses)
{
    std::lock_guard<std::mutex> lock(m_render_lock);
    m_poses = poses;
}

void Visualization::set_points(const std::vector<Eigen::Vector3f>& points)
{
    std::lock_guard<std::mutex> lock(m_render_lock);
    m_points = points;
}

void Visualization::draw_points(const std::vector<Eigen::Vector3f>& points)
{
    glPointSize(2);
    glBegin(GL_POINTS);
    glColor3f(1.0f, 1.0f, 1.0f);
    for (const auto& point : points) {
        glVertex3f(point[0], point[1], point[2]);
    }
    glEnd();
}

void Visualization::run()
{
    if (!m_initialized) {
        throw std::runtime_error("Visualization not initialized");
    }
    pangolin::BindToContext(m_window_name);

    pangolin::RegisterKeyPressCallback(pangolin::PANGO_KEY_TAB, [&]() {
        std::lock_guard<std::mutex> lock(m_key_pressed_mutex);
        m_key_pressed_cv.notify_all();
    });

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        m_3d_display->Activate(*m_camera_state);

        // Draw coordinate frame
        glLineWidth(3);
        pangolin::glDrawAxis(3.0);

        {
            std::lock_guard<std::mutex> lock(m_render_lock);
            for (const auto& pose : m_poses) {
                draw_camera_pose(pose);
            }
            draw_points(m_points);

            m_image_display->Activate();
            if (!m_image.empty()) {
                if (!m_image_texture) {
                    m_image_texture = std::make_unique<pangolin::GlTexture>(m_image.cols,
                                                                            m_image.rows,
                                                                            GL_RGB,
                                                                            true,
                                                                            0,
                                                                            GL_RGB,
                                                                            GL_UNSIGNED_BYTE);
                    m_image_texture->Upload(m_image.data, GL_RGB, GL_UNSIGNED_BYTE);
                }

                glColor3f(1.0f, 1.0f, 1.0f);
                m_image_texture->RenderToViewport(true);
            }
        }

        pangolin::FinishFrame();
    }

    pangolin::DestroyWindow(m_window_name);
    {
        std::lock_guard<std::mutex> lock(m_key_pressed_mutex);
        m_key_pressed_cv.notify_all();
    }
    m_has_quit = true;
}

bool Visualization::has_quit() const { return m_has_quit; }

void Visualization::run_threaded()
{
    std::thread([this]() { run(); }).detach();
}

} // namespace slam