cmake_minimum_required(VERSION 3.10)
project(slam VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Add source files
add_executable(slam src/main.cpp src/video.cpp src/visual_odom.cpp src/graphics.cpp src/map.cpp)

target_link_libraries(slam ${OpenCV_LIBS} OpenGL::GL OpenGL::GLU glfw glm::glm)
