#pragma once
#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <vector>
#include <mutex>

namespace slam {

class Visualization {
public:
    Visualization(const std::string& window_name = "3D Viewer");
    ~Visualization();
    
    void set_camera_poses(const std::vector<Eigen::Matrix4f>& poses);
    void set_points(const std::vector<Eigen::Vector3f>& points);

    void run();
    
private:
    void draw_camera_pose(const Eigen::Matrix4f& pose);
    void draw_points(const std::vector<Eigen::Vector3f>& points);
    void initialize(int width = 1024, int height = 768);
    
    std::string m_window_name;
    pangolin::OpenGlRenderState* m_camera_state;
    pangolin::Handler3D* m_handler;
    pangolin::View* m_display;
    bool m_initialized;

    // Data to render
    std::mutex m_data_mutex;
    std::vector<Eigen::Matrix4f> m_poses;
    std::vector<Eigen::Vector3f> m_points;
};

} // namespace slam