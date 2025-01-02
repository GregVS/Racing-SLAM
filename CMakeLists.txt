cmake_minimum_required(VERSION 3.20)
project(slam VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Packages
find_package(OpenCV REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(g2o REQUIRED)

file(GLOB SRC_FILES src/*.cpp)
add_executable(slam ${SRC_FILES})

set(G2O_LIBRARIES
    g2o_core
    g2o_stuff
    g2o_types_slam3d
    g2o_types_sba
    g2o_solver_eigen
)

target_link_libraries(slam ${OpenCV_LIBS} ${OPENGL_LIBRARIES} glfw glm::glm Eigen3::Eigen ${G2O_LIBRARIES} fmt::fmt)
