#include <iostream>
#include <glm/gtc/matrix_transform.hpp>

#include "Graphics.h"

namespace slam
{
    
Scene scene_from_map(const Map& map)
{
    Scene scene;
    for (const auto& frame : map.get_frames()) {
        scene.poses.push_back(frame->get_pose());
    }
    for (const auto& [_, point] : map.get_map_points()) {
        scene.map_points.push_back(point.get_position());
    }
    return scene;
}

Graphics::Graphics(int w, int h)
        : m_width(w)
        , m_height(h)
{
    m_camera_pos = glm::vec3(3.0f, 20.0f, 3.0f);
    m_camera_front = glm::vec3(0.0f, 0.0f, -1.0f);
    m_camera_up = glm::vec3(0.0f, -1.0f, 0.0f);
    m_yaw = 90.0f;
    m_pitch = -60.0f;
    m_first_mouse = true;
    m_last_x = m_width / 2.0f;
    m_last_y = m_height / 2.0f;
    m_mouse_button_pressed = false;
}

Graphics::~Graphics()
{
    glfwTerminate();
}

void Graphics::initialize_gl()
{
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    m_window = glfwCreateWindow(m_width, m_height, "3D Viewer", NULL, NULL);
    if (!m_window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(m_window);
    glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);
    glfwSetCursorPosCallback(m_window, mouse_callback);
    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetMouseButtonCallback(m_window, mouse_button_callback);
    glfwSetWindowUserPointer(m_window, this);

    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        throw std::runtime_error("Failed to initialize GLAD");
    }

    glEnable(GL_DEPTH_TEST);
    glViewport(0, 0, m_width, m_height);
}

bool Graphics::is_running()
{
    return !glfwWindowShouldClose(m_window);
}

void draw_camera()
{
    glBegin(GL_TRIANGLES);
    glVertex3f(-0.5f, 0.0f, 0.0f);
    glVertex3f(0.5f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.5f);
    glEnd();
}

void draw_camera_poses(const Scene &scene)
{
    for (const auto &pose : scene.poses) {
        glPushMatrix();

        double openGLMatrix[16] = { pose.at<double>(0, 0), pose.at<double>(1, 0), pose.at<double>(2, 0), 0.0,
                                    pose.at<double>(0, 1), pose.at<double>(1, 1), pose.at<double>(2, 1), 0.0,
                                    pose.at<double>(0, 2), pose.at<double>(1, 2), pose.at<double>(2, 2), 0.0,
                                    pose.at<double>(0, 3), pose.at<double>(1, 3), pose.at<double>(2, 3), 1.0 };
        glMultMatrixd(openGLMatrix);

        draw_camera();
        glPopMatrix();
    }
}

void draw_point_cloud(const Scene &scene)
{
    glBegin(GL_POINTS);
    glColor3f(1.0f, 1.0f, 1.0f);
    glPointSize(3.0f);
    for (const auto& point : scene.map_points) {
        glVertex3f(point.x, point.y, point.z);
    }
    glEnd();
}

void Graphics::set_scene(const Scene& scene) 
{
    std::lock_guard<std::mutex> guard(m_scene_lock);
    m_scene = scene;
}

void Graphics::run() {
    initialize_gl();
    while (is_running()) {
        update();
    }
}

void Graphics::update()
{
    process_input();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Set up view/projection matrices
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)m_width / m_height, 0.1f, 1000.0f);
    glLoadMatrixf(&projection[0][0]);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glm::mat4 view = glm::lookAt(m_camera_pos, m_camera_pos + m_camera_front, m_camera_up);
    glLoadMatrixf(&view[0][0]);

    {
        std::lock_guard<std::mutex> guard(m_scene_lock);
        draw_axes();
        draw_camera_poses(m_scene);
        draw_point_cloud(m_scene);
    }

    glfwSwapBuffers(m_window);
    glfwPollEvents();
}

void Graphics::draw_axes()
{
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    // X axis (red)
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(1.0f, 0.0f, 0.0f);

    // Y axis (green)
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 1.0f, 0.0f);

    // Z axis (blue)
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 1.0f);
    glEnd();
}

void Graphics::process_input()
{
    const float camera_speed = 0.2f;

    glm::vec3 right = glm::normalize(glm::cross(m_camera_front, m_camera_up));

    // Forward/Backward
    if (glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS)
        m_camera_pos += camera_speed * m_camera_front;
    if (glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS)
        m_camera_pos -= camera_speed * m_camera_front;

    // Left/Right
    if (glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS)
        m_camera_pos -= camera_speed * right;
    if (glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS)
        m_camera_pos += camera_speed * right;

    // Up/Down
    if (glfwGetKey(m_window, GLFW_KEY_SPACE) == GLFW_PRESS)
        m_camera_pos += camera_speed * m_camera_up;
    if (glfwGetKey(m_window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
        m_camera_pos -= camera_speed * m_camera_up;

    if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(m_window, true);
}

void Graphics::framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void Graphics::mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    Graphics *graphics = static_cast<Graphics *>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_RIGHT) {
        graphics->m_mouse_button_pressed = (action == GLFW_PRESS);
        if (graphics->m_mouse_button_pressed) {
            graphics->m_first_mouse = true;
        }
    }
}

void Graphics::mouse_callback(GLFWwindow *window, double xpos, double ypos)
{
    Graphics *graphics = static_cast<Graphics *>(glfwGetWindowUserPointer(window));

    if (!graphics->m_mouse_button_pressed) {
        return;
    }

    if (graphics->m_first_mouse) {
        graphics->m_last_x = xpos;
        graphics->m_last_y = ypos;
        graphics->m_first_mouse = false;
        return;
    }

    float xoffset = xpos - graphics->m_last_x;
    float yoffset = graphics->m_last_y - ypos;
    graphics->m_last_x = xpos;
    graphics->m_last_y = ypos;

    const float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        // Pan camera
        graphics->m_camera_pos -=
                xoffset * 0.05f * glm::normalize(glm::cross(graphics->m_camera_front, graphics->m_camera_up));
        graphics->m_camera_pos -= yoffset * 0.05f * graphics->m_camera_up;
    } else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
        // Rotate camera
        graphics->m_yaw += xoffset;
        graphics->m_pitch += yoffset;

        if (graphics->m_pitch > 89.0f)
            graphics->m_pitch = 89.0f;
        if (graphics->m_pitch < -89.0f)
            graphics->m_pitch = -89.0f;

        // Update camera front vector
        glm::vec3 front;
        front.x = cos(glm::radians(graphics->m_yaw)) * cos(glm::radians(graphics->m_pitch));
        front.y = sin(glm::radians(graphics->m_pitch));
        front.z = sin(glm::radians(graphics->m_yaw)) * cos(glm::radians(graphics->m_pitch));
        graphics->m_camera_front = glm::normalize(front);
    }
}

};