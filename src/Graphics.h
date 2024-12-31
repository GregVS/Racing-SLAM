#pragma once
#include <opencv2/opencv.hpp>
#include <GLFW/glfw3.h>
#include <GL/glu.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "Map.h"

namespace slam
{

class Graphics {
public:
    Graphics(int width = 800, int height = 600);

    ~Graphics();
    
    bool is_running();

    void draw_scene(const std::vector<cv::Mat> &camera_poses, const Map &map);
    
private:
    GLFWwindow* m_window;
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
    
    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);

    static void mouse_callback(GLFWwindow* window, double xpos, double ypos);

    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
};


};