#include "MapMatching.h"
#include "Camera.h"
#include <vector>

namespace slam
{

void match_map_points(Map &map, Frame &frame)
{
    int matches = 0;
    for (auto &[_, point] : map.get_map_points()) {
        auto point2D = map.get_camera().to_image_coordinates(point.get_position(), frame.get_pose());

        // Check if point is in frame
        if (!within_frame(point2D, map.get_camera()) || point.is_observed_by(&frame)) {
            continue;
        }

        // Find a match
        auto keypoint_indices = frame.get_keypoints_within_radius(point2D, 2.0f);
        for (const auto keypoint : keypoint_indices) {
            if (frame.get_corresponding_map_point(keypoint)) {
                continue;
            }

            float orbDist = map_point_orb_distance(point, frame.get_descriptor(keypoint));
            if (orbDist < 32) {
                point.add_observation(&frame, keypoint);
                frame.set_corresponding_map_point(keypoint, &point);
                matches++;
                break;
            }
        }
    }

    std::cout << "Map Point Matches: " << matches << std::endl;
}

void piggyback_prev_frame_matches(Map &map, const Frame &prev_frame, Frame &frame,
                                  const std::vector<cv::DMatch> &matches)
{
    int matched = 0;
    for (const auto &match : matches) {
        if (frame.get_corresponding_map_point(match.queryIdx)) {
            continue;
        }

        MapPoint *mapPoint = prev_frame.get_corresponding_map_point(match.trainIdx);
        if (mapPoint) {
            mapPoint->add_observation(&frame, match.queryIdx);
            frame.set_corresponding_map_point(match.queryIdx, mapPoint);
            matched++;
        }
    }
    std::cout << "Piggybacked matches: " << matched << std::endl;
}

void triangulate_points(Map &map, Frame &prev_frame, Frame &frame, const std::vector<cv::DMatch> &matches)
{
    std::vector<cv::Point2f> from_points, to_points;
    std::vector<int> frame_keypoint_indices;
    std::vector<int> prev_frame_keypoint_indices;
    for (const auto &match : matches) {
        if (prev_frame.get_corresponding_map_point(match.trainIdx) ||
            frame.get_corresponding_map_point(match.queryIdx)) {
            continue;
        }

        from_points.push_back(prev_frame.get_keypoint(match.trainIdx).pt);
        to_points.push_back(frame.get_keypoint(match.queryIdx).pt);
        frame_keypoint_indices.push_back(match.queryIdx);
        prev_frame_keypoint_indices.push_back(match.trainIdx);
    }

    // Compute the projection matrices
    cv::Mat P1 = map.get_camera().get_projection_matrix(prev_frame.get_pose());
    cv::Mat P2 = map.get_camera().get_projection_matrix(frame.get_pose());

    // Triangulate
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, from_points, to_points, points4D);

    // Convert the points to 3D and compute reprojection errors
    int addedPoints = 0;
    std::vector<float> reprojection_errors;

    for (int i = 0; i < points4D.cols; i++) {
        cv::Point3f point3D;
        point3D.x = points4D.at<float>(0, i) / points4D.at<float>(3, i);
        point3D.y = points4D.at<float>(1, i) / points4D.at<float>(3, i);
        point3D.z = points4D.at<float>(2, i) / points4D.at<float>(3, i);

        // Check if point is in front of camera
        auto point_in_camera_1 = map.get_camera().to_camera_coordinates(point3D, prev_frame.get_pose());
        auto point_in_camera_2 = map.get_camera().to_camera_coordinates(point3D, frame.get_pose());
        if (point_in_camera_1.z < 0 || point_in_camera_2.z < 0) {
            continue;
        }

        // Compute reprojection errors
        auto reprojection1 = map.get_camera().to_image_coordinates(point3D, prev_frame.get_pose());
        auto reprojection2 = map.get_camera().to_image_coordinates(point3D, frame.get_pose());

        float err1 = cv::norm(reprojection1 - prev_frame.get_keypoint(prev_frame_keypoint_indices[i]).pt);
        float err2 = cv::norm(reprojection2 - frame.get_keypoint(frame_keypoint_indices[i]).pt);

        if (err1 > 2.0 || err2 > 2.0) {
            continue;
        }

        reprojection_errors.push_back(err1);
        reprojection_errors.push_back(err2);

        auto &mapPoint = map.add_map_point(point3D);
        mapPoint.add_observation(&prev_frame, prev_frame_keypoint_indices[i]);
        prev_frame.set_corresponding_map_point(prev_frame_keypoint_indices[i], &mapPoint);
        mapPoint.add_observation(&frame, frame_keypoint_indices[i]);
        frame.set_corresponding_map_point(frame_keypoint_indices[i], &mapPoint);
        addedPoints++;
    }

    std::cout << "Triangulated points: " << addedPoints << std::endl;
    if (addedPoints > 0) {
        std::cout << "Triangulation reprojection error: " << cv::mean(reprojection_errors)[0] << std::endl;
    }
}

float reprojection_error(const Map &map)
{
    float error = 0;
    int count = 0;

    Frame* lastFrame = map.get_frames().back().get();

    for (const auto &[_, point] : map.get_map_points()) {
        point.for_each_observation([&](Frame *frame, int keypointIndex) {
            auto point2D = map.get_camera().to_image_coordinates(point.get_position(), frame->get_pose());
            auto err = cv::norm(point2D - frame->get_keypoint(keypointIndex).pt);
            error += err;
            count++;
        });
    }
    return error / count;
}

void cull_points(Map &map)
{
    const float MAX_REPROJECTION_ERROR = 2.0;
    const int MIN_OBSERVATIONS = 3;
    const int MIN_OBSERVATIONS_GRACE_PERIOD = 10;
    
    int removedPoints = 0;
    std::vector<int> idsToCull;
    for (const auto &[_, point] : map.get_map_points()) {
        float error = 0;
        int count = 0;
        point.for_each_observation([&](Frame *frame, int keypointIndex) {
            auto point2D = map.get_camera().to_image_coordinates(point.get_position(), frame->get_pose());
            auto err = cv::norm(point2D - frame->get_keypoint(keypointIndex).pt);
            error += err;
            count++;
        });
        bool seen_recently = point.get_observations_vector().back().first->get_id() >= (map.get_next_frame_id() - MIN_OBSERVATIONS_GRACE_PERIOD);
        if (error / count > MAX_REPROJECTION_ERROR) {
            std::cout << "Culling point " << point.get_id() << " with reprojection error " << error / count << std::endl;
            idsToCull.push_back(point.get_id());
        } else if (point.observation_count() < MIN_OBSERVATIONS && !seen_recently) {
            idsToCull.push_back(point.get_id());
        }
    }

    for (const auto &id : idsToCull) {
        auto &point = map.get_map_point(id);
        point.for_each_observation(
                [&](Frame *frame, int keypointIndex) { frame->set_corresponding_map_point(keypointIndex, nullptr); });
        map.remove_map_point(point.get_id());
        removedPoints++;
    }
    std::cout << "Culled " << removedPoints << " points" << std::endl;
}

};