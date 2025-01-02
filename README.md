# Video-SLAM
Monocular video SLAM implementation in C++.

## Libraries
- Uses OpenCV for ORB feature extraction, matching, and triangulation
- OpenGL for rendering 3D point cloud and poses

## Roadmap:
- [X] Feature extraction and matching
- [X] Point cloud triangulation and matching
- [X] Essential matrix estimation with RANSAC
- [X] Bundle adjustment/pose graph optimization with g2o
- [ ] Optimize performance to prevent slowdown with more points
- [ ] Run the 3d vis on separate thread

## Dependencies
Dependencies are installed using vcpkg. It requires env var `VCPKG_ROOT` to be set.

To configure: `cmake --preset=vcpkg` and to build `cmake --build build`
