#pragma once

#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <vector>
#include <mutex>
#include <memory>
#include <condition_variable>

#include "Frame.h"

namespace slam {

class Visualization {
public:
    Visualization(const std::string& window_name = "3D Viewer");
    ~Visualization();

    void set_camera_poses(const std::vector<Eigen::Matrix4f>& poses);
    void set_points(const std::vector<Eigen::Vector3f>& points);
    void set_image(const cv::Mat& image);

    void initialize(int width = 1024, int height = 1024);
    void run();
    void run_threaded();
    void wait_for_keypress();
    bool has_quit() const;
    
private:
    void draw_camera_pose(const Eigen::Matrix4f& pose);
    void draw_points(const std::vector<Eigen::Vector3f>& points);
    
    std::string m_window_name;
    pangolin::OpenGlRenderState* m_camera_state = nullptr;
    pangolin::Handler3D* m_handler = nullptr;
    pangolin::View* m_3d_display = nullptr;
    pangolin::View* m_image_display = nullptr;
    bool m_initialized = false;

    std::atomic<bool> m_has_quit = false;

    // Data to render
    std::mutex m_render_lock;
    std::vector<Eigen::Matrix4f> m_poses;
    std::vector<Eigen::Vector3f> m_points;
    cv::Mat m_image;
    std::unique_ptr<pangolin::GlTexture> m_image_texture;

    std::mutex m_key_pressed_mutex;
    std::condition_variable m_key_pressed_cv;
};

} // namespace slam