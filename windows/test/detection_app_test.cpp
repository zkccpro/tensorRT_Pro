#include <gtest/gtest.h>

#include <detection.h>
#include <pre_processing.h>

class DetAppCase : public ::testing::Test {
protected:
    virtual void SetUp() {
        engine = CREATE_MMDEPLOY_PLUGIN_DET_INFER("workspace/faster_rcnn_mmdeploy.trt");
        image_OK = cv::imread("workspace/OK_origin.jpg");
        image_NG = cv::imread("workspace/NG_origin.jpg");
        for (int i = 0; i < engine->immutable_infer()->get_max_batch_size(); ++i) {
            images_OK.push_back(image_OK);
            images_NG.push_back(image_NG);
        }
    }
    // 记得初始化哦...
    std::shared_ptr<App::Engine<Detection::DetResult>> engine;
    std::vector<cv::Mat> images_OK;
    std::vector<cv::Mat> images_NG;
    cv::Mat image_OK;
    cv::Mat image_NG;

    std::array<float, 3> mean{123.675f, 116.28f, 103.53f};
    std::array<float, 3> std{58.395f, 57.12f, 57.375f};
};

TEST_F(DetAppCase, RunSinglePicture) {
    ASSERT_NE(engine, nullptr);
    auto result = engine->run(image_OK, mean, std);
}

TEST_F(DetAppCase, RunMultiPictures) {
    ASSERT_NE(engine, nullptr);
    engine->run(images_NG, mean, std);
}

TEST_F(DetAppCase, SinglePictureCPUPreProcessing) {
    ASSERT_NE(engine, nullptr);
    auto height = engine->immutable_infer()->input(0)->height();
    auto width = engine->immutable_infer()->input(0)->width();
    auto f = preprocessing::resize_keep_aspect_ratio(image_OK, cv::Size{height, width}, image_OK);
    FMT_INFO("%f %f", f.width, f.height);
    FMT_INFO("%d %d", image_OK.size().width, image_OK.size().height);
    auto result = engine->run(image_OK, mean, std); // 当然这里会再次resize到模型指定大小的
}

// TEST_F(DetAppCase, SinglePictureGPUPreProcessing) {

// }

TEST_F(DetAppCase, SingleResultCPUParse) {
    ASSERT_NE(engine, nullptr);

    auto result = engine->run(image_NG, mean, std);
    ASSERT_NE(result->defect_num(), 0);
    
    FMT_INFO(result->format());
    cv::imwrite("/zkcc_workspace/zkccpro/tensorRT_Pro/workspace/DetAppCase.SingleResultParse__result_format.jpg", result->format(image_NG));
}

TEST_F(DetAppCase, MultiResultCPUParse) {
    ASSERT_NE(engine, nullptr);

    auto results = engine->run(images_NG, mean, std);

    for (int i = 0; i < results.size(); ++i) {
        FMT_INFO(results[i]->format());
        cv::imwrite(iLogger::string_format("/zkcc_workspace/zkccpro/tensorRT_Pro/workspace/DetAppCase.MultiResultCPUParse__result_format_%d.jpg", i), results[i]->format((images_NG)[i]));
    }
}

// TEST_F(DetAppCase, SingleResultGPUParse) {
    
// }

// TEST_F(DetAppCase, MultiResultGPUParse) {
    
// }


/// 期望正常执行的边界测例
TEST_F(DetAppCase, EmptyInputImage) {
    ASSERT_NE(engine, nullptr);

    cv::Mat image;
    // 期望在这里弹出警告，返回空指针
    auto result = engine->run(image, mean, std);
    ASSERT_EQ(result, nullptr);
}

TEST_F(DetAppCase, EmptyMultiInputImage) {
    ASSERT_NE(engine, nullptr);
    for (auto& image : images_NG) {
        image = cv::Mat();
    }
    // 无论有多少个空图片也不应该core掉
    // 把空图片赋值为相应尺寸的全0图片，并警告
    auto results = engine->run(images_NG, mean, std);

    for (int i = 0; i < results.size(); ++i) {
        FMT_INFO(results[i]->format());
        cv::imwrite(iLogger::string_format("/zkcc_workspace/zkccpro/tensorRT_Pro/workspace/DetAppCase.MultiResultCPUParse__result_format_%d.jpg", i), results[i]->format((images_NG)[i]));
    }
}

TEST_F(DetAppCase, DetResultAPI) {
    ASSERT_NE(engine, nullptr);

    auto results = engine->run(images_NG, mean, std);

    for (auto& result : results) {
        INFO("---------Current Result DetResultAPI TEST----------");
        // get immutable defects API
        std::vector<Detection::BBox> im_defects = result->immutable_defects();
        ASSERT_EQ(im_defects.size(), result->defect_num());
        int origin_bbox_num = result->defect_num();
        FMT_INFO("origin_bbox_num is: %d", origin_bbox_num);

        int add_num {5};
        for (int i = 0; i < add_num; ++i) {
            im_defects.emplace_back(1, 2, 3, 4, 5, 6);
        }
        ASSERT_EQ(origin_bbox_num, result->defect_num());

        im_defects.clear();
        ASSERT_EQ(origin_bbox_num, result->defect_num());

        // get mutable defects API
        std::vector<Detection::BBox>& defects = result->mutable_defects();
        ASSERT_EQ(defects.size(), result->defect_num());
        origin_bbox_num = result->defect_num();
        FMT_INFO("origin_bbox_num is: %d", origin_bbox_num);

        add_num = 10;
        for (int i = 0; i < add_num; ++i) {
            defects.emplace_back(1, 2, 3, 4, 5, 6);
        }
        ASSERT_EQ(defects.size(), result->defect_num());
        ASSERT_EQ(origin_bbox_num + add_num, result->defect_num());
        FMT_INFO("cur bbox num is: %d", result->defect_num());

        defects.clear();
        ASSERT_EQ(defects.size(), result->defect_num());
        ASSERT_EQ(0, result->defect_num());
        ASSERT_EQ(true, result->ok());
        FMT_INFO("cur bbox num is: %d", result->defect_num());
    }
}

/// 期望assert掉的边界测例

// TEST_F(DetAppCase, InvalidInferPath) {
//     // 期望在这里直接ASSERT掉
//     auto empty_engine = App::create_infer<Detection::DetResult>("/zkcc_workspace/model/trt/wrong.trt", Detection::faster_rcnn_parser);
//     INFO("SHOULD NOT BE HERE");
//     auto result = empty_engine->run(image_OK, mean, std);
// }

// TEST_F(DetAppCase, OverNumberMultiInputImage) {
//     ASSERT_NE(engine, nullptr);
//     images_NG->push_back(cv::Mat(2, 2, CV_8UC3, cv::Scalar(0, 0, 0)));
//     // 期望在这里志记并core掉
//     engine->run(images_NG, mean, std);
// }

// TEST_F(DetAppCase, EmptyFormatInputImage) {
//     ASSERT_NE(engine, nullptr);

//     auto result = engine->run(image_NG, mean, std);
//     cv::Mat empty_img;
//     // 期望在format函数里给出警告
//     cv::imwrite("/zkcc_workspace/zkccpro/tensorRT_Pro/workspace/DetAppCase.SingleResultParse__result_format.jpg", result->format(empty_img));
// }