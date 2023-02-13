#include <gtest/gtest.h>

#include <detection.h>
#include <pre_processing.h>

class DetAppCase : public ::testing::Test {
protected:
    virtual void SetUp() {
        initLibAmirstanInferPlugins();
        engine = App::create_infer<Detection::DetResult>("workspace/faster_rcnn_batch=1_trt7.trt", Detection::faster_rcnn_parser);
        image_OK = std::make_shared<cv::Mat>(cv::imread("workspace/OK_origin.jpg"));
        image_NG = std::make_shared<cv::Mat>(cv::imread("workspace/NG_origin.jpg"));
        
        images_OK = std::make_shared<std::vector<cv::Mat>>();
        images_NG = std::make_shared<std::vector<cv::Mat>>();
        for (int i = 0; i < engine->get_infer()->get_max_batch_size(); ++i) {
            images_OK->push_back(*image_OK);
            images_NG->push_back(*image_NG);
        }
    }
    // 记得初始化哦...
    std::shared_ptr<App::Engine<Detection::DetResult>> engine;
    std::shared_ptr<std::vector<cv::Mat>> images_OK;
    std::shared_ptr<std::vector<cv::Mat>> images_NG;
    std::shared_ptr<cv::Mat> image_OK;
    std::shared_ptr<cv::Mat> image_NG;
};

TEST_F(DetAppCase, RunSinglePicture) {
    ASSERT_NE(engine, nullptr);

    auto result = engine->run(*image_OK);
}

TEST_F(DetAppCase, RunMultiPictures) {
    ASSERT_NE(engine, nullptr);
    engine->run(*images_NG);
}

TEST_F(DetAppCase, SinglePictureCPUPreProcessing) {
    ASSERT_NE(engine, nullptr);

    cv::Mat img = preprocessing::basic_prepro_cpu(*image_OK, {2016, 2016});
    auto result = engine->run(img);
}

// TEST_F(DetAppCase, SinglePictureGPUPreProcessing) {

// }

TEST_F(DetAppCase, FasterRCNNSingleResultCPUParse) {
    ASSERT_NE(engine, nullptr);

    auto result = engine->run(*image_NG);
    ASSERT_NE(result->defect_num(), 0);
    
    FMT_INFO(result->format());
    cv::imwrite("workspace/DetAppCase.FasterRCNNSingleResultParse__result_format.jpg", result->format(*image_NG));
}

TEST_F(DetAppCase, FasterRCNNMultiResultCPUParse) {
    ASSERT_NE(engine, nullptr);

    auto results = engine->run(*images_NG);

    for (int i = 0; i < results.size(); ++i) {
        FMT_INFO(results[i]->format());
        cv::imwrite(iLogger::string_format("workspace/DetAppCase.FasterRCNNMultiResultCPUParse__result_format_%d.jpg", i), results[i]->format((*images_NG)[i]));
    }
}

// TEST_F(DetAppCase, FasterRCNNSingleResultGPUParse) {
    
// }

// TEST_F(DetAppCase, FasterRCNNMultiResultGPUParse) {
    
// }


/// 期望assert掉的边界测例

// TEST_F(DetAppCase, InvalidInferPath) {
//     // 期望在这里直接ASSERT掉
//     auto empty_engine = App::create_infer<Detection::DetResult>("/zkcc_workspace/model/trt/wrong.trt", Detection::faster_rcnn_parser);
//     INFO("SHOULD NOT BE HERE");
//     auto result = empty_engine->run(*image_OK);
// }

// TEST_F(DetAppCase, OverNumberMultiInputImage) {
//     ASSERT_NE(engine, nullptr);
//     images_NG->push_back(cv::Mat(2, 2, CV_8UC3, cv::Scalar(0, 0, 0)));
//     // 期望在这里志记并core掉
//     engine->run(*images_NG);
// }

// TEST_F(DetAppCase, EmptyFormatInputImage) {
//     ASSERT_NE(engine, nullptr);

//     auto result = engine->run(*image_NG);
//     cv::Mat empty_img;
//     // 期望在format函数里给出警告
//     cv::imwrite("/zkcc_workspace/zkccpro/tensorRT_Pro/workspace/DetAppCase.FasterRCNNSingleResultParse__result_format.jpg", result->format(empty_img));
// }


/// 期望正常执行的边界测例
TEST_F(DetAppCase, EmptyInputImage) {
    ASSERT_NE(engine, nullptr);

    cv::Mat image;
    // 期望在这里弹出警告，返回空指针
    auto result = engine->run(image);
    ASSERT_EQ(result, nullptr);
}

TEST_F(DetAppCase, EmptyMultiInputImage) {
    ASSERT_NE(engine, nullptr);
    for (auto& image : *images_NG) {
        image = cv::Mat();
    }
    // 无论有多少个空图片也不应该core掉
    // 把空图片赋值为相应尺寸的全0图片，并警告
    auto results = engine->run(*images_NG);

    for (int i = 0; i < results.size(); ++i) {
        FMT_INFO(results[i]->format());
        cv::imwrite(iLogger::string_format("/zkcc_workspace/zkccpro/tensorRT_Pro/workspace/DetAppCase.FasterRCNNMultiResultCPUParse__result_format_%d.jpg", i), results[i]->format((*images_NG)[i]));
    }
}
