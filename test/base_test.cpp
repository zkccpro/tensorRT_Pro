#include <gtest/gtest.h>

#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <opencv2/opencv.hpp>
// #include <plugin/amirInferPlugin.h>
#include <memory>
#include <vector>

TEST(BaseCase, InitEngine) {
    // initLibAmirstanInferPlugins();
    auto engine = TRT::load_infer("/zkcc_workspace/model/trt/faster_rcnn_epoch_10.trt");
    ASSERT_NE(engine, nullptr);
    // print model info
    engine->print();
}

TEST(BaseCase, Forward) {
    //initLibAmirstanInferPlugins();
    auto engine = TRT::load_infer("/zkcc_workspace/model/trt/faster_rcnn_epoch_10.trt");
    ASSERT_NE(engine, nullptr);
    // print model info
    engine->print();

    auto image = cv::imread("/zkcc_workspace/data/data_cizhuan/coco/train2017/20220310124752-origin.jpg");
    cv::resize(image, image, cv::Size(2016, 2016));

    // get the model input and output node, which can be accessed by name or index
    auto input = engine->input(0);   // or auto input = engine->input("images");
    ASSERT_NE(input, nullptr);
    auto output = engine->output(0); // or auto output = engine->output("output");
    ASSERT_NE(output, nullptr);
    // put the image into input tensor by calling set_norm_mat()
    float mean[] = {0, 0, 0};
    float std[]  = {1, 1, 1};
    input->set_norm_mat(0, image, mean, std);

    // do the inference. Here sync(true) or async(false) is optional
    engine->forward(); // engine->forward(true or false)

    // get the outut_ptr, which can used to access the output
    int* output_ptr = output->cpu<int>();
    ASSERT_NE(output_ptr, nullptr);
    std::cout << *output_ptr << std::endl;
}

TEST(BaseCase, All) {
    //initLibAmirstanInferPlugins();
    auto engine = TRT::load_infer("/zkcc_workspace/model/trt/faster_rcnn_epoch_10.trt");

    // print model info
    engine->print();

    // load image
    auto image = cv::imread("/zkcc_workspace/data/data_cizhuan/coco/train2017/20220310124752-origin.jpg");
    cv::resize(image, image, cv::Size(2016, 2016));

    // get the model input and output node, which can be accessed by name or index
    auto input = engine->input(0);   // or auto input = engine->input("images");
    std::cout << engine->num_output() << std::endl;
    auto output = engine->output(1); // or auto output = engine->output("output");

    // put the image into input tensor by calling set_norm_mat()
    float mean[] = {0, 0, 0};
    float std[]  = {1, 1, 1};
    input->set_norm_mat(0, image, mean, std);

    // do the inference. Here sync(true) or async(false) is optional
    engine->forward(); // engine->forward(true or false)

    // get the outut_ptr, which can used to access the output
    // int* output_ptr = output->cpu<int>();
    // std::cout << *output_ptr << std::endl;
    auto dims = input->dims();
    for (auto dim : dims) {
        std::cout << dim << std::endl;
    }
    std::cout << input->at<float>(0,2,3,4) << std::endl;
}
