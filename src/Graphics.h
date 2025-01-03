#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>
#include <mutex>

#include "Map.h"

namespace slam
{

struct Scene {
    std::vector<cv::Mat> poses;
    std::vector<cv::Point3f> map_points;
};

Scene scene_from_map(const Map& map);

class Graphics {
public:
    Graphics(int width = 800, int height = 600);

    ~Graphics();

    bool is_running();

    void run();

    void set_scene(const Scene& scene);

private:
    GLFWwindow *m_window;
    Scene m_scene;
    std::mutex m_scene_lock;

    int m_width, m_height;

    // Camera
    glm::vec3 m_camera_pos;
    glm::vec3 m_camera_front;
    glm::vec3 m_camera_up;
    float m_yaw;
    float m_pitch;

    // Mouse handling
    bool m_first_mouse;
    float m_last_x, m_last_y;
    bool m_mouse_button_pressed;

    void initialize_gl();
    void draw_axes();
    void process_input();

    void update();

    static void framebuffer_size_callback(GLFWwindow *window, int width, int height);

    static void mouse_callback(GLFWwindow *window, double xpos, double ypos);

    static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
};

};