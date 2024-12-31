cmake_minimum_required(VERSION 3.20)
project(slam VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Packages
find_package(OpenCV REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

file(GLOB SRC_FILES src/*.cpp)

add_executable(slam ${SRC_FILES})

target_link_libraries(slam ${OpenCV_LIBS} ${OPENGL_LIBRARIES} glfw glm::glm)
