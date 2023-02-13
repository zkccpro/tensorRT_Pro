#include "pre_processing.h"

namespace preprocessing {
    cv::Mat basic_prepro_cpu(const cv::Mat& input, const std::vector<int>& output_size) {
        cv::Mat output;
        cv::resize(input, output, cv::Size(output_size[0], output_size[1]));
        return output;
    }
}; // preprocessing