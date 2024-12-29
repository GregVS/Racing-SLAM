#include "map.h"
#include "video.h"
#include "visual_odom.h"
#include "graphics.h"

int main()
{
    std::string videoPath = "snow_video.mp4";

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
        cv::Mat image;

        std::vector<cv::Mat> cameraPoses;
        Map map;
        nextFrame(cap, image);

        auto start = std::chrono::high_resolution_clock::now();

        while (graphics.isRunning()) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            if (duration.count() > 200) {
                start = end;
                nextFrame(cap, image);

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
                drawMatches(prevFrame, frame, matches);

                auto pose = estimatePose(prevFrame, frame, matches);
                frame.setPose(pose);
                cameraPoses.push_back(pose);

                matchMapPoints(map, frame, camera);
                triangulatePoints(map, prevFrame, frame, matches);

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
