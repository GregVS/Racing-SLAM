#pragma once

#include <Eigen/Core>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <vector>

namespace slam {

class Visualization {
  public:
    struct Point {
        Eigen::Vector3f position;
        cv::Vec3b color;
    };

    Visualization(const std::string& window_name = "3D Viewer");
    ~Visualization();

    void push_frame(size_t frame_index,
                    const cv::Mat& image,
                    const std::vector<Eigen::Matrix4f>& poses,
                    const std::vector<Point>& points,
                    const std::vector<Eigen::Vector3f>& culled,
                    const std::string& status,
                    double meters_per_unit = 0.0);
    bool wait_for_step();
    void initialize(int width = 1024, int height = 1024);
    void run();
    void run_threaded();
    void wait_for_keypress();
    bool has_quit() const;

  private:
    struct Snapshot {
        size_t frame_index = 0;
        cv::Mat image;
        std::vector<Eigen::Matrix4f> poses;
        std::vector<Point> points;
        std::vector<Eigen::Vector3f> culled;
        std::string status;
        double meters_per_unit = 0.0; // zero before IMU alignment
    };

    void draw_camera_poses(const Snapshot& snapshot);
    void draw_points(const Snapshot& snapshot);
    void draw_image(const Snapshot& snapshot);
    void step_view(int delta);
    void toggle_pause();

    // Pangolin
    std::string m_window_name;
    pangolin::OpenGlRenderState* m_camera_state = nullptr;
    pangolin::Handler3D* m_handler = nullptr;
    pangolin::View* m_3d_display = nullptr;
    pangolin::View* m_image_display = nullptr;
    bool m_initialized = false;

    std::atomic<bool> m_has_quit = false;

    static constexpr size_t MAX_HISTORY = 300; // for replay
    std::mutex m_render_lock;
    std::deque<Snapshot> m_history;
    size_t m_view_offset = 0; // frames behind the newest, 0 when live
    std::unique_ptr<pangolin::GlTexture> m_image_texture;
    bool m_texture_stale = true;

    std::mutex m_step_mutex;
    std::condition_variable m_step_cv;
    std::atomic<bool> m_paused = false;
    bool m_step_requested = false;

    std::mutex m_key_pressed_mutex;
    std::condition_variable m_key_pressed_cv;
};

} // namespace slam