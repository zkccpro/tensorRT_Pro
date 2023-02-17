
#include <opencv2/opencv.hpp>
#include <vector>

namespace preprocessing {
    cv::Size2f resize_keep_aspect_ratio(const cv::Mat& input, const cv::Size& dst_size, cv::Mat& output);
}; // preprocessing
