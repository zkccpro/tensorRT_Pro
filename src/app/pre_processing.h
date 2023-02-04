
#include <opencv2/opencv.hpp>
#include <vector>

namespace preprocessing {
    cv::Mat basic_prepro_cpu(const cv::Mat& input, const std::vector<int>& output_size);
}; // preprocessing