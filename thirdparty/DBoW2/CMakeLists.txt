cmake_minimum_required(VERSION 3.10)
project(DBoW2)

add_library(DBoW2 STATIC
    DBoW2/BowVector.cpp
    DBoW2/FORB.cpp
    DBoW2/FeatureVector.cpp
    DBoW2/ScoringObject.cpp
    DUtils/Random.cpp
    DUtils/Timestamp.cpp)
target_include_directories(DBoW2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/DBoW2
    ${CMAKE_CURRENT_SOURCE_DIR}/DUtils)
target_link_libraries(DBoW2 PUBLIC ${OpenCV_LIBS})
