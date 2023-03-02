#include <gtest/gtest.h>

#include <detection.h>
#include <pre_processing.h>


class BenchMark : public ::testing::Test {
protected:
    virtual void SetUp() {
        image_NG = cv::imread("workspace/NG_origin.jpg");

        trt8_engine = CREATE_AMIRSTAN_PLUGIN_DET_INFER("workspace/faster_rcnn_batch=8_trt8.trt");
        // multibatch_trt8_engine = CREATE_MMDEPLOY_PLUGIN_DET_INFER("workspace/faster_rcnn_mmdeploy.trt");
        // for (int i = 0; i < multibatch_trt8_engine->immutable_infer()->get_max_batch_size(); ++i) {
        //     images_NG.push_back(image_NG);
        // }
    }
    // 记得初始化哦...
    std::shared_ptr<App::Engine<Detection::DetResult>> trt8_engine;
    std::shared_ptr<App::Engine<Detection::DetResult>> multibatch_trt8_engine;

    cv::Mat image_NG;
    std::vector<cv::Mat> images_NG;

    std::array<float, 3> mean{123.675f, 116.28f, 103.53f};
    std::array<float, 3> std{58.395f, 57.12f, 57.375f};
};

// trt8单batch测试
TEST_F(BenchMark, EmptyTest) {
    ASSERT_NE(trt8_engine, nullptr);
}
TEST_F(BenchMark, TRT8SingleImage) {
    ASSERT_NE(trt8_engine, nullptr);

    for (int i = 0; i < 500; ++i) {
        auto result = trt8_engine->run(image_NG, mean, std);
        result->format();
    }
}

// trt8多batch测试
// TEST_F(BenchMark, EmptyTest) {
//     ASSERT_NE(multibatch_trt8_engine, nullptr);
// }
TEST_F(BenchMark, TRT8MultiImages) {
    ASSERT_NE(multibatch_trt8_engine, nullptr);

    for (int i = 0; i < 500; ++i) {
        auto result = multibatch_trt8_engine->run(images_NG, mean, std);
        for (auto& ret : result) {
            ret->format();
        }
    }
}
