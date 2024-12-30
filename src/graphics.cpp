#include "graphics.h"
#include <iostream>

Graphics::Graphics(int w, int h) : width(w), height(h) {
    cameraPos = glm::vec3(3.0f, 3.0f, 3.0f);
    cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    cameraUp = glm::vec3(0.0f, -1.0f, 0.0f);
    yaw = -135.0f;
    pitch = -45.0f;
    firstMouse = true;
    lastX = width / 2.0f;
    lastY = height / 2.0f;
    mouseButtonPressed = false;
    
    initializeGL();
}

Graphics::~Graphics() {
    glfwTerminate();
}

void Graphics::initializeGL() {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    window = glfwCreateWindow(width, height, "3D Viewer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetWindowUserPointer(window, this);
    
    glEnable(GL_DEPTH_TEST);
    glViewport(0, 0, width, height);
}

bool Graphics::isRunning() {
    return !glfwWindowShouldClose(window);
}

void drawCamera() {
    glBegin(GL_TRIANGLES);
    glVertex3f(-0.5f, 0.0f, 0.0f);
    glVertex3f(0.5f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.5f);
    glEnd();
}

void drawCameraPoses(const std::vector<cv::Mat> &cameraPoses) {
    for (const auto& pose : cameraPoses) {
        glPushMatrix();

        double openGLMatrix[16] = {
            pose.at<double>(0, 0), pose.at<double>(1, 0),
            pose.at<double>(2, 0), 0.0,
            pose.at<double>(0, 1), pose.at<double>(1, 1),
            pose.at<double>(2, 1), 0.0,
            pose.at<double>(0, 2), pose.at<double>(1, 2),
            pose.at<double>(2, 2), 0.0,
            pose.at<double>(0, 3), pose.at<double>(1, 3),
            pose.at<double>(2, 3), 1.0 
        };
        glMultMatrixd(openGLMatrix);

        drawCamera();
        glPopMatrix();
    }
}

void drawPointCloud(const Map &map) {
    glBegin(GL_POINTS);
    glColor3f(1.0f, 1.0f, 1.0f);
    for (const auto& [_, point] : map.getMapPoints()) {
        glVertex3f(point.getPosition().x, point.getPosition().y, point.getPosition().z);
    }
    glEnd();
}

void Graphics::drawScene(const std::vector<cv::Mat> &cameraPoses, const Map &map) {
    processInput();
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    
    // Set up view/projection matrices
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (float)width/height, 0.1f, 1000.0f);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    glLoadMatrixf(&view[0][0]);
    
    drawAxes();
    drawCameraPoses(cameraPoses);
    drawPointCloud(map);

    glfwSwapBuffers(window);
    glfwPollEvents();
}

void Graphics::drawAxes() {
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

void Graphics::processInput() {
    const float cameraSpeed = 0.1f;
    
    glm::vec3 right = glm::normalize(glm::cross(cameraFront, cameraUp));
    
    // Forward/Backward
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;
        
    // Left/Right
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= cameraSpeed * right;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += cameraSpeed * right;
    
    // Up/Down
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraUp;
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraUp;
    
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void Graphics::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void Graphics::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    Graphics* graphics = static_cast<Graphics*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_RIGHT) {
        graphics->mouseButtonPressed = (action == GLFW_PRESS);
        if (graphics->mouseButtonPressed) {
            graphics->firstMouse = true;
        }
    }
}

void Graphics::mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    Graphics* graphics = static_cast<Graphics*>(glfwGetWindowUserPointer(window));
    
    if (!graphics->mouseButtonPressed) {
        return;
    }
    
    if (graphics->firstMouse) {
        graphics->lastX = xpos;
        graphics->lastY = ypos;
        graphics->firstMouse = false;
        return;
    }
    
    float xoffset = xpos - graphics->lastX;
    float yoffset = graphics->lastY - ypos;
    graphics->lastX = xpos;
    graphics->lastY = ypos;
    
    const float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;
    
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        // Pan camera
        graphics->cameraPos -= xoffset * 0.05f * glm::normalize(glm::cross(graphics->cameraFront, graphics->cameraUp));
        graphics->cameraPos -= yoffset * 0.05f * graphics->cameraUp;
    }
    else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
        // Rotate camera
        graphics->yaw += xoffset;
        graphics->pitch += yoffset;
        
        if (graphics->pitch > 89.0f)
            graphics->pitch = 89.0f;
        if (graphics->pitch < -89.0f)
            graphics->pitch = -89.0f;
        
        // Update camera front vector
        glm::vec3 front;
        front.x = cos(glm::radians(graphics->yaw)) * cos(glm::radians(graphics->pitch));
        front.y = sin(glm::radians(graphics->pitch));
        front.z = sin(glm::radians(graphics->yaw)) * cos(glm::radians(graphics->pitch));
        graphics->cameraFront = glm::normalize(front);
    }
}