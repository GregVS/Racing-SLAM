#pragma once
#include <opencv2/opencv.hpp>
#include <GLFW/glfw3.h>
#include <GL/glu.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "visual_odom.h"

class Graphics {
public:
    Graphics(int width = 800, int height = 600);
    ~Graphics();
    
    bool isRunning();
    void drawScene(const std::vector<cv::Mat> &cameraPoses, const Map &map);
    
private:
    GLFWwindow* window;
    int width, height;
    
    // Camera
    glm::vec3 cameraPos;
    glm::vec3 cameraFront;
    glm::vec3 cameraUp;
    float yaw;
    float pitch;
    
    // Mouse handling
    bool firstMouse;
    float lastX, lastY;
    bool mouseButtonPressed;
    
    void initializeGL();
    void drawAxes();
    void processInput();
    
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
};

