#include <gtest/gtest.h>

#include <detection.h>

class DetAppCase : public ::testing::Test {
protected:
    virtual void SetUp() {
        initLibAmirstanInferPlugins();
        engine = App::create_infer<Detection::DetResult>("/zkcc_workspace/model/trt/faster_rcnn_epoch_10.trt", Detection::faster_rcnn_parser);
        image_OK = std::make_shared<cv::Mat>(cv::imread("/zkcc_workspace/zkccpro/tensorRT_Pro/workspace/OK_origin.jpg"));
        image_NG = std::make_shared<cv::Mat>(cv::imread("/zkcc_workspace/zkccpro/tensorRT_Pro/workspace/NG_origin.jpg"));
    }

    std::shared_ptr<App::Engine<Detection::DetResult>> engine { nullptr };
    std::shared_ptr<cv::Mat> image_OK;
    std::shared_ptr<cv::Mat> image_NG;
};

TEST_F(DetAppCase, RunSinglePicture) {
    ASSERT_NE(engine, nullptr);

    auto result = engine->run(*image_OK);
}

// TEST_F(DetAppCase, RunMultiPicture) {

// }

TEST_F(DetAppCase, FasterRCNNSingleResultCPUParse) {
    ASSERT_NE(engine, nullptr);

    auto result = engine->run(*image_NG);
    ASSERT_NE(result.defect_num(), 0);
    
    INFO(result.format().c_str());
    cv::imwrite("/zkcc_workspace/zkccpro/tensorRT_Pro/workspace/DetAppCase.FasterRCNNSingleResultParse__result_format.jpg", result.format(*image_NG));
}

// TEST_F(DetAppCase, FasterRCNNMultiResultCPUParse) {
    
// }

// TEST_F(DetAppCase, FasterRCNNSingleResultGPUParse) {
    
// }

// TEST_F(DetAppCase, FasterRCNNMultiResultGPUParse) {
    
// }

/// 边界测例

// TEST_F(DetAppCase, InvalidInferPath) {
//     // 期望在这里直接ASSERT掉
//     auto empty_engine = App::create_infer<Detection::DetResult>("/zkcc_workspace/model/trt/wrong.trt", Detection::faster_rcnn_parser);
//     INFO("NOT BE HERE");
//     auto result = empty_engine->run(*image_OK);
// }

// TEST_F(DetAppCase, EmptyInputImage) {
//     ASSERT_NE(engine, nullptr);

//     cv::Mat image;
//     // 期望在这里直接ASSERT掉
//     auto result = engine->run(image);
// }

// TEST_F(DetAppCase, EmptyMultiInputImage) {

// }

TEST_F(DetAppCase, EmptyFormatInputImage) {
    ASSERT_NE(engine, nullptr);

    auto result = engine->run(*image_NG);
    cv::Mat empty_img;
    // 期望在format函数里给出警告
    cv::imwrite("/zkcc_workspace/zkccpro/tensorRT_Pro/workspace/DetAppCase.FasterRCNNSingleResultParse__result_format.jpg", result.format(empty_img));
}
