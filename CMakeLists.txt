cmake_minimum_required(VERSION 3.10)

# Project name and version
project(slam VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Add source files
add_executable(slam src/main.cpp src/video.cpp)

# Link OpenCV libraries
target_link_libraries(slam ${OpenCV_LIBS})
