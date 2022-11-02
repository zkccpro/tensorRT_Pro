#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <opencv2/opencv.hpp>

int main() {
    // // convert onnx to trt
    // TRT::compile(
    //     TRT::Mode::FP32,            // compile model in fp32
    //     1,                          // max batch size
    //     "/zkcc_workspace/model/onnx/faster_rcnn.onnx",              // onnx file
    //     "/zkcc_workspace/model/trt/faster_rcnn_epoch_49.trt",     // save path
    //     {{{1, 3, 2016, 2016}}}                          //  redefine the shape of input when needed
    // );

    // load model and get a shared_ptr. get nullptr if fail to load.
    auto engine = TRT::load_infer("/zkcc_workspace/model/trt/mobilenet.trt");

    // print model info
    engine->print();

    // load image
    auto image = cv::imread("/zkcc_workspace/data/data_cizhuan/coco/train2017/20220310124752-origin.jpg");
    cv::resize(image, image, cv::Size(2016, 2016));

    // get the model input and output node, which can be accessed by name or index
    auto input = engine->input(0);   // or auto input = engine->input("images");
    auto output = engine->output(0); // or auto output = engine->output("output");

    // put the image into input tensor by calling set_norm_mat()
    float mean[] = {0, 0, 0};
    float std[]  = {1, 1, 1};
    input->set_norm_mat(0, image, mean, std);

    // do the inference. Here sync(true) or async(false) is optional
    engine->forward(); // engine->forward(true or false)

    // get the outut_ptr, which can used to access the output
    float* output_ptr = output->cpu<float>();
    std::cout << *output_ptr << std::endl;

    return 0;
}

