cmake_minimum_required(VERSION 3.10)
project(slam VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Packages
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Eigen3 REQUIRED)

file(GLOB SRC_FILES src/*.cpp)
list(REMOVE_ITEM SRC_FILES */main.cpp)
add_executable(slam ${SRC_FILES} src/main.cpp)
add_library(slam_lib ${SRC_FILES})
target_link_libraries(slam_lib Eigen3::Eigen Ceres::ceres ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
target_link_libraries(slam Eigen3::Eigen Ceres::ceres ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

file(GLOB TEST_FILES test/*.cpp)
foreach(TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)  # Get name without extension
    add_executable(${TEST_NAME} ${TEST_FILE})
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(${TEST_NAME} PRIVATE slam_lib Eigen3::Eigen Ceres::ceres ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
endforeach()

