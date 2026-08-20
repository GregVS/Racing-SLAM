#include "Visualization.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <pangolin/display/process.h>
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
    m_image_display = &pangolin::CreateDisplay().SetBounds(0.0, 0.4, 0.0, 0.65);
    m_top_down_display = &pangolin::CreateDisplay().SetBounds(0.0, 0.4, 0.65, 1.0);

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

float grid_step(float extent)
{
    const float target = std::max(extent, 1e-3F) / 10.0F;
    const float magnitude = std::pow(10.0F, std::floor(std::log10(target)));
    const float normalized = target / magnitude;
    if (normalized < 1.5F) {
        return magnitude;
    }
    if (normalized < 3.5F) {
        return 2.0F * magnitude;
    }
    if (normalized < 7.5F) {
        return 5.0F * magnitude;
    }
    return 10.0F * magnitude;
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

// Top down XZ view of the trajectory, X right and Z up on screen
void Visualization::draw_top_down(const Snapshot& snapshot)
{
    m_top_down_display->Activate();
    if (snapshot.poses.empty()) {
        return;
    }

    const float scale = visual_scale(snapshot.meters_per_unit);

    std::vector<Eigen::Vector2f> track;
    track.reserve(snapshot.poses.size());
    Eigen::Vector2f heading(0.0F, 1.0F);
    for (const Eigen::Matrix4f& pose : snapshot.poses) {
        const Eigen::Matrix3f rotation = pose.block<3, 3>(0, 0);
        const Eigen::Vector3f centre = -rotation.transpose() * pose.block<3, 1>(0, 3) * scale;
        track.emplace_back(centre[0], centre[2]);
        heading = Eigen::Vector2f(rotation(2, 0), rotation(2, 2));
    }

    Eigen::Vector2f minimum = track.front();
    Eigen::Vector2f maximum = track.front();
    for (const Eigen::Vector2f& position : track) {
        minimum = minimum.cwiseMin(position);
        maximum = maximum.cwiseMax(position);
    }

    const Eigen::Vector2f centre = 0.5F * (minimum + maximum);
    float half_extent = 0.5F * (maximum - minimum).maxCoeff();
    half_extent = std::max(half_extent, 1.0F) * 1.15F;

    const float aspect = m_top_down_display->v.h > 0
                             ? static_cast<float>(m_top_down_display->v.w) / static_cast<float>(m_top_down_display->v.h)
                             : 1.0F;
    const float half_width = aspect >= 1.0F ? half_extent * aspect : half_extent;
    const float half_height = aspect >= 1.0F ? half_extent : half_extent / aspect;

    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(
        centre[0] - half_width, centre[0] + half_width, centre[1] - half_height, centre[1] + half_height, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // Grid
    const float step = grid_step(2.0F * half_extent);
    const float left = std::floor((centre[0] - half_width) / step) * step;
    const float right = centre[0] + half_width;
    const float bottom = std::floor((centre[1] - half_height) / step) * step;
    const float top = centre[1] + half_height;
    glLineWidth(1);
    glColor3f(0.22f, 0.22f, 0.22f);
    glBegin(GL_LINES);
    for (float x = left; x <= right; x += step) {
        glVertex2f(x, centre[1] - half_height);
        glVertex2f(x, centre[1] + half_height);
    }
    for (float y = bottom; y <= top; y += step) {
        glVertex2f(centre[0] - half_width, y);
        glVertex2f(centre[0] + half_width, y);
    }
    glEnd();

    // Map points
    glPointSize(2);
    glBegin(GL_POINTS);
    for (const auto& point : snapshot.points) {
        glColor3ub(point.color[0] / 2, point.color[1] / 2, point.color[2] / 2);
        const Eigen::Vector3f position = point.position * scale;
        glVertex2f(position[0], position[2]);
    }
    glEnd();

    // Trajectory
    glLineWidth(2);
    glColor3f(0.3f, 0.6f, 1.0f);
    glBegin(GL_LINE_STRIP);
    for (const Eigen::Vector2f& position : track) {
        glVertex2f(position[0], position[1]);
    }
    glEnd();

    // Current pose with heading
    const Eigen::Vector2f current = track.back();
    const float marker = 0.02F * half_extent * 2.0F;
    if (heading.norm() > 1e-6F) {
        heading.normalize();
        glLineWidth(2);
        glColor3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_LINES);
        glVertex2f(current[0], current[1]);
        glVertex2f(current[0] + heading[0] * marker * 4.0F, current[1] + heading[1] * marker * 4.0F);
        glEnd();
    }
    glPointSize(8);
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_POINTS);
    glVertex2f(current[0], current[1]);
    glEnd();

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_DEPTH_TEST);
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
                draw_top_down(snapshot);
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