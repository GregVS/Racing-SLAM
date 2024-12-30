#include "video.h"
#include "visual_odom.h"
#include "graphics.h"
#include "Features.h"

int main()
{
    std::string videoPath = "videos/highway.mp4";

    try {
        cv::VideoCapture cap = initializeVideo(videoPath);

        Graphics graphics(800, 600);

        int W = 1920;
        int H = 1080;
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        K.at<double>(0, 0) = 800;
        K.at<double>(1, 1) = 800;
        K.at<double>(0, 2) = W / 2;
        K.at<double>(1, 2) = H / 2;

        Camera camera{ K, W, H };

        std::vector<cv::Mat> cameraPoses;
        Map map(camera);

        auto start = std::chrono::high_resolution_clock::now();

        while (graphics.isRunning()) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            if (duration.count() > 200) {
                start = end;
                cv::Mat image = nextFrame(cap);

                // Process frame
                if (map.getFrames().empty()) {
                    Frame &frame = map.addFrame(extractFeatures(image, map.getNextFrameId()));
                    frame.setPose(cv::Mat::eye(4, 4, CV_64F));
                    cameraPoses.push_back(frame.getPose());
                    continue;
                }

                Frame &prevFrame = map.getLastFrame();
                Frame &frame = map.addFrame(extractFeatures(image, map.getNextFrameId()));

                auto matches = matchFeatures(prevFrame, frame);
                auto poseEstimate = estimatePose(prevFrame, frame, camera, matches);
                drawMatches(prevFrame, frame, poseEstimate.filteredMatches);

                frame.setPose(poseEstimate.pose);
                cameraPoses.push_back(poseEstimate.pose);

                matchMapPoints(map, frame);
                triangulatePoints(map, prevFrame, frame, poseEstimate.filteredMatches);

                std::cout << "Number of map points: " << map.getMapPoints().size() << std::endl;

                cv::waitKey(1);
            }

            // Draw 3d scene
            graphics.drawScene(cameraPoses, map);
        }

        cap.release();
        cv::destroyAllWindows();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
