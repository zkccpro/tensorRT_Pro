#include "pre_processing.h"
#include <ilogger.hpp>

cv::Size2f preprocessing::resize_keep_aspect_ratio(const cv::Mat& input, const cv::Size& dst_size, cv::Mat& output) {
    //const cv::Mat& temp_input = input;  // keep header (required if the function was called with same input and output)
    const int input_cols = input.cols;
    const int input_rows = input.rows;
    float h = dst_size.width  * (input.rows / (float) input.cols);
    float w = dst_size.height * (input.cols / (float) input.rows);
    if( h <= dst_size.height) {
        w = dst_size.width;
    } else {
        h = dst_size.height;
    }
    cv::resize(input, output, cv::Size(w, h));
    float fx = (float) output.cols / input_cols;
    float fy = (float) output.rows / input_rows;
    cv::Size2d scale_factor(fx, fy);
    int top  = 0;
    int left = 0;
    int bottom = dst_size.height - output.rows;
    int right  = dst_size.width  - output.cols;
    cv::copyMakeBorder(output, output, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
    return scale_factor;
}
