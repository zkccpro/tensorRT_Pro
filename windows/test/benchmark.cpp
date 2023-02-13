#include <gtest/gtest.h>

#include <detection.h>
#include <pre_processing.h>


class BenchMark : public ::testing::Test {
protected:
    virtual void SetUp() {
        initLibAmirstanInferPlugins();

        image_NG = std::make_shared<cv::Mat>(cv::imread("workspace/NG_origin.jpg"));
        images_NG = std::make_shared<std::vector<cv::Mat>>();

         //trt7_engine = App::create_infer<Detection::DetResult>("workspace/faster_rcnn_batch=1_trt7.trt", Detection::faster_rcnn_parser);
         //multibatch_trt7_engine = App::create_infer<Detection::DetResult>("workspace/faster_rcnn_batch=8_trt7.trt", Detection::faster_rcnn_parser);
         //for (int i = 0; i < multibatch_trt7_engine->get_infer()->get_max_batch_size(); ++i) {
         //    images_NG->push_back(*image_NG);
         //}

        trt8_engine = App::create_infer<Detection::DetResult>("workspace/faster_rcnn_batch=1_trt8.trt", Detection::faster_rcnn_parser);
        multibatch_trt8_engine = App::create_infer<Detection::DetResult>("workspace/faster_rcnn_batch=8_trt8.trt", Detection::faster_rcnn_parser);
        for (int i = 0; i < multibatch_trt8_engine->get_infer()->get_max_batch_size(); ++i) {
            images_NG->push_back(*image_NG);
        }
    }
    // 记得初始化哦...
    std::shared_ptr<App::Engine<Detection::DetResult>> trt7_engine;
    std::shared_ptr<App::Engine<Detection::DetResult>> trt8_engine;

    std::shared_ptr<App::Engine<Detection::DetResult>> multibatch_trt7_engine;
    std::shared_ptr<App::Engine<Detection::DetResult>> multibatch_trt8_engine;

    std::shared_ptr<cv::Mat> image_NG;

    std::shared_ptr<std::vector<cv::Mat>> images_NG;
};

 //// trt7单batch测试
 //TEST_F(BenchMark, EmptyTest) {
 //    ASSERT_NE(trt7_engine, nullptr);
 //}
 //TEST_F(BenchMark, TRT7SingleImage) {
 //    ASSERT_NE(trt7_engine, nullptr);

 //    for (int i = 0; i < 500; ++i) {
 //        auto result = trt7_engine->run(*image_NG);
 //        result->format();
 //    }
 //}

 //// trt7多batch测试
 //TEST_F(BenchMark, TRT7MultiImages) {
 //    ASSERT_NE(multibatch_trt7_engine, nullptr);

 //    for (int i = 0; i < 500; ++i) {
 //        auto result = multibatch_trt7_engine->run(*images_NG);
 //        for (auto& ret : result) {
 //            ret->format();
 //        }
 //    }
 //}

// trt8单batch测试
TEST_F(BenchMark, EmptyTest) {
    ASSERT_NE(trt8_engine, nullptr);
}
TEST_F(BenchMark, TRT8SingleImage) {
    ASSERT_NE(trt8_engine, nullptr);

    for (int i = 0; i < 500; ++i) {
        auto result = trt8_engine->run(*image_NG);
        result->format();
    }
}

// trt8多batch测试
TEST_F(BenchMark, TRT8MultiImages) {
    ASSERT_NE(multibatch_trt8_engine, nullptr);

    for (int i = 0; i < 500; ++i) {
        auto result = multibatch_trt8_engine->run(*images_NG);
        for (auto& ret : result) {
            ret->format();
        }
    }
}
