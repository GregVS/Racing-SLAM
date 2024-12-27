#include "video.h"
#include "visual_odom.h"
#include "graphics.h"

int main()
{
    std::string videoPath = "snow_video.mp4";

    try {
        cv::VideoCapture cap = initializeVideo(videoPath);

        Graphics graphics(800, 600);

        cv::Mat frame;
        FrameData prevFrameData;

        std::vector<cv::Mat> cameraPoses;

        auto start = std::chrono::high_resolution_clock::now();

        while (graphics.isRunning()) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            if (duration.count() > 100) {
                start = end;
                nextFrame(cap, frame);

                // Process frame
                FrameData frameData = extractFeatures(frame);
                if (!prevFrameData.frame.empty()) {
                    auto matches = matchFeatures(prevFrameData, frameData);
                    drawMatches(prevFrameData, frameData, matches);

                    auto pose = estimatePose(prevFrameData, frameData, matches);
                    frameData.pose = pose;
                    cameraPoses.push_back(pose);
                } else {
                    // Initial camera pose
                    frameData.pose = cv::Mat::eye(4, 4, CV_64F);
                }
                prevFrameData = frameData;
                cv::waitKey(1);
            }

            // Draw 3d scene
            graphics.drawScene(cameraPoses);
        }

        cap.release();
        cv::destroyAllWindows();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
