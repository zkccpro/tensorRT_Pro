#include <gtest/gtest.h>

#include <detection.h>

class PluginParserCase : public ::testing::Test {
protected:
    virtual void SetUp() {
        engine = CREATE_MMDEPLOY_PLUGIN_DET_INFER("faster_rcnn_mmdeploy.trt");
    }
    // 记得初始化哦...
    std::shared_ptr<App::Engine<Detection::DetResult>> engine;
};

TEST_F(PluginParserCase, InvalidOutputNum) {
    ASSERT_NE(engine, nullptr);

    auto& infer = engine->mutable_infer();
    int num_output = infer->num_output();
    std::vector<std::shared_ptr<TRT::Tensor>> output;
    output.reserve(num_output);
    for (int i = 0; i < num_output - 1; ++i) {
        output.push_back(infer->output(i));
    }

    Detection::mmdeploy_det_plg_parser->check_valid(output);
}

TEST_F(PluginParserCase, InvalidDataType) {
    ASSERT_NE(engine, nullptr);

    auto& infer = engine->mutable_infer();
    int num_output = infer->num_output();
    std::vector<std::shared_ptr<TRT::Tensor>> output;
    output.reserve(num_output);
    for (int i = 0; i < num_output; ++i) {
        auto tensor = infer->output(i);
        output.push_back(std::make_shared<TRT::Tensor>(TRT::DataType::Unknow, tensor->get_data()));
    }

    Detection::mmdeploy_det_plg_parser->check_valid(output);
}

TEST_F(PluginParserCase, InvalidNdims) {
    ASSERT_NE(engine, nullptr);

    auto& infer = engine->mutable_infer();
    int num_output = infer->num_output();
    std::vector<std::shared_ptr<TRT::Tensor>> output;
    output.reserve(num_output);
    for (int i = 0; i < num_output; ++i) {
        auto tensor = infer->output(i);
        output.push_back(std::make_shared<TRT::Tensor>(1, std::array<int, 1>{1}.data()));
    }

    Detection::mmdeploy_det_plg_parser->check_valid(output);
}

TEST_F(PluginParserCase, InvalidShape) {
    ASSERT_NE(engine, nullptr);

    auto& infer = engine->mutable_infer();
    int num_output = infer->num_output();
    std::vector<std::shared_ptr<TRT::Tensor>> output;
    output.reserve(num_output);
    for (int i = 0; i < num_output; ++i) {
        auto tensor = infer->output(i);
        tensor->resize_single_dim(1, 10);
        output.push_back(tensor);
    }
    
    Detection::mmdeploy_det_plg_parser->check_valid(output);
}
