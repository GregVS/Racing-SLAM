#include "video.h"

int main()
{
    std::string videoPath = "stock_video.mp4";

    try {
        cv::VideoCapture cap = initializeVideo(videoPath);
        playVideo(cap);

        cap.release();
        cv::destroyAllWindows();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
