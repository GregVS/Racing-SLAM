cmake_minimum_required(VERSION 3.10)
project(slam VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Packages
find_package(OpenCV REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glad REQUIRED)
find_package(g2o REQUIRED)

file(GLOB SRC_FILES src/*.cpp)
add_executable(slam ${SRC_FILES})

set(G2O_LIBRARIES
    g2o::core
    g2o::stuff
    g2o::types_slam3d
    g2o::types_sba
    g2o::solver_eigen
)

target_link_libraries(slam glm::glm glad::glad glfw ${G2O_LIBRARIES} ${OpenCV_LIBS})
