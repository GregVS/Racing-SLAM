#include "Visualization.h"

#include <chrono>
#include <pangolin/pangolin.h>
#include <thread>

namespace slam {

Visualization::Visualization(const std::string& window_name) : m_window_name(window_name) {}

Visualization::~Visualization()
{
    delete m_camera_state;
    delete m_handler;
    // m_display should be deleted by Pangolin
}

void Visualization::initialize(int width, int height)
{
    pangolin::CreateWindowAndBind(m_window_name, width, height);
    glEnable(GL_DEPTH_TEST);

    m_camera_state = new pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(width, height, 420, 420, width / 2, height / 2, 0.2, 4000),
        pangolin::ModelViewLookAt(-2, -2, -2, 0, 0, 5, pangolin::AxisNegY));

    m_handler = new pangolin::Handler3D(*m_camera_state, pangolin::AxisNegY);
    m_3d_display = &pangolin::CreateDisplay().SetBounds(0.4, 1.0, 0.0, 1.0).SetHandler(m_handler);
    m_image_display = &pangolin::CreateDisplay().SetBounds(0.0, 0.4, 0.0, 1.0);

    pangolin::GetBoundWindow()->RemoveCurrent();
    m_initialized = true;
    m_has_quit = false;
}

void Visualization::wait_for_keypress()
{
    std::unique_lock<std::mutex> lock(m_key_pressed_mutex);
    m_key_pressed_cv.wait(lock);
}

bool Visualization::has_quit() const
{
    return m_has_quit;
}

void Visualization::run_threaded()
{
    std::thread([this]() { run(); }).detach();
}

namespace {

float visual_scale(double meters_per_unit)
{
    return meters_per_unit > 0.0 ? static_cast<float>(1.0 / meters_per_unit) : 1.0F;
}

} // namespace

void Visualization::push_frame(size_t frame_index,
                               const cv::Mat& image,
                               const std::vector<Eigen::Matrix4f>& poses,
                               const std::vector<Point>& points,
                               const std::vector<Eigen::Vector3f>& culled,
                               const std::string& status,
                               double meters_per_unit)
{
    Snapshot snapshot;
    snapshot.frame_index = frame_index;
    cv::cvtColor(image, snapshot.image, cv::COLOR_BGR2RGB);
    snapshot.poses = poses;
    snapshot.points = points;
    snapshot.culled = culled;
    snapshot.status = status;
    snapshot.meters_per_unit = meters_per_unit;

    std::lock_guard<std::mutex> lock(m_render_lock);
    m_history.push_back(std::move(snapshot));
    while (m_history.size() > MAX_HISTORY) {
        m_history.pop_front();
        if (m_view_offset > 0 && m_view_offset >= m_history.size()) {
            m_view_offset = m_history.size() - 1;
        }
    }
    m_texture_stale = true;
}

void Visualization::step_view(int delta)
{
    {
        std::lock_guard<std::mutex> lock(m_render_lock);
        if (delta < 0) {
            if (m_view_offset + 1 < m_history.size()) {
                m_view_offset++;
                m_texture_stale = true;
            }
            return;
        }
        if (m_view_offset > 0) {
            m_view_offset--;
            m_texture_stale = true;
            return;
        }
    }
    {
        std::lock_guard<std::mutex> lock(m_step_mutex);
        m_step_requested = true;
    }
    m_step_cv.notify_all();
}

void Visualization::toggle_pause()
{
    {
        std::lock_guard<std::mutex> lock(m_step_mutex);
        m_paused = !m_paused;
    }
    m_step_cv.notify_all();
}

bool Visualization::wait_for_step()
{
    std::unique_lock<std::mutex> lock(m_step_mutex);
    m_step_cv.wait(lock, [this]() { return !m_paused || m_step_requested || m_has_quit; });
    m_step_requested = false;
    return !m_has_quit;
}

void Visualization::draw_camera_poses(const Snapshot& snapshot)
{
    const float scale = visual_scale(snapshot.meters_per_unit);
    const size_t n = snapshot.poses.size();
    for (size_t i = 0; i < n; ++i) {
        const float camera_size = 1.5f;
        Eigen::Matrix4f pose = snapshot.poses[i];
        pose.block<3, 1>(0, 3) *= scale;
        const Eigen::Matrix4f inverse_pose = pose.inverse();

        glPushMatrix();
        glMultMatrixf(inverse_pose.data());

        if (i == n - 1) {
            glColor3f(0.0f, 1.0f, 0.0f); // Draw last (current) camera in green
        } else {
            glColor3f(0.0f, 0.0f, 1.0f); // Draw others in blue
        }
        glBegin(GL_TRIANGLES);
        glVertex3f(0, 0, 0);
        glVertex3f(camera_size, 0, -camera_size);
        glVertex3f(-camera_size, 0, -camera_size);
        glEnd();

        glPopMatrix();
    }
}

void Visualization::draw_points(const Snapshot& snapshot)
{
    const float scale = visual_scale(snapshot.meters_per_unit);
    glPointSize(3);
    glBegin(GL_POINTS);
    for (const auto& point : snapshot.points) {
        glColor3ub(point.color[0], point.color[1], point.color[2]);
        const Eigen::Vector3f position = point.position * scale;
        glVertex3f(position[0], position[1], position[2]);
    }
    glEnd();

    glPointSize(6);
    glBegin(GL_POINTS);
    glColor3ub(255, 0, 0);
    for (const auto& position : snapshot.culled) {
        const Eigen::Vector3f scaled = position * scale;
        glVertex3f(scaled[0], scaled[1], scaled[2]);
    }
    glEnd();
}

void Visualization::draw_image(const Snapshot& snapshot)
{
    m_image_display->Activate();
    if (snapshot.image.empty()) {
        return;
    }

    cv::Mat labelled = snapshot.image.clone();
    std::string label = "frame " + std::to_string(snapshot.frame_index);
    if (m_view_offset > 0) {
        label += " [replay -" + std::to_string(m_view_offset) + "]";
    } else if (m_paused) {
        label += " [paused]";
    }
    cv::putText(labelled, label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    if (!snapshot.status.empty()) {
        auto colour = snapshot.culled.empty() ? cv::Scalar(255, 255, 255) : cv::Scalar(255, 80, 80);
        cv::putText(labelled, snapshot.status, cv::Point(10, 58), cv::FONT_HERSHEY_SIMPLEX, 0.6, colour, 2);
    }

    if (!m_image_texture || m_texture_stale) {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        m_image_texture = std::make_unique<pangolin::GlTexture>(
            labelled.cols, labelled.rows, GL_RGB, true, 0, GL_RGB, GL_UNSIGNED_BYTE);
        m_texture_stale = false;
    }
    m_image_texture->Upload(labelled.data, GL_RGB, GL_UNSIGNED_BYTE);

    glColor3f(1.0f, 1.0f, 1.0f);
    m_image_texture->RenderToViewport(true);
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

    pangolin::RegisterKeyPressCallback('p', [this]() { toggle_pause(); });
    pangolin::RegisterKeyPressCallback(' ', [this]() { toggle_pause(); });
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_SPECIAL + pangolin::PANGO_KEY_RIGHT, [this]() {
        m_paused = true;
        step_view(1);
    });
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_SPECIAL + pangolin::PANGO_KEY_LEFT, [this]() {
        m_paused = true;
        step_view(-1);
    });
    pangolin::RegisterKeyPressCallback('.', [this]() {
        m_paused = true;
        step_view(1);
    });
    pangolin::RegisterKeyPressCallback(',', [this]() {
        m_paused = true;
        step_view(-1);
    });

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        m_3d_display->Activate(*m_camera_state);

        // Draw coordinate frame
        glLineWidth(3);
        pangolin::glDrawAxis(3.0);

        {
            std::lock_guard<std::mutex> lock(m_render_lock);
            if (!m_history.empty()) {
                const Snapshot& snapshot = m_history[m_history.size() - 1 - m_view_offset];
                draw_camera_poses(snapshot);
                draw_points(snapshot);
                draw_image(snapshot);
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
    m_step_cv.notify_all();
}

} // namespace slam