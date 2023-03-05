

#ifndef APP_HPP
#define APP_HPP

#include <string>
#include <vector>
#include <array>
#include <memory>

// #include <plugin/amirInferPlugin.h>
#include <infer/trt_infer.hpp>
#include <opencv2/opencv.hpp>
#include <ilogger.hpp>

namespace App {
    #define CREATE_AMIRSTAN_PLUGIN_DET_INFER(path) App::create_infer<Detection::DetResult>(path, std::dynamic_pointer_cast<App::BaseParser<Detection::DetResult>>(Detection::amirstan_det_plg_parser))
    #define CREATE_MMDEPLOY_PLUGIN_DET_INFER(path) App::create_infer<Detection::DetResult>(path, std::dynamic_pointer_cast<App::BaseParser<Detection::DetResult>>(Detection::mmdeploy_det_plg_parser))

    class Result {
    public:
        virtual ~Result() = default;
        virtual std::string format() = 0;
        virtual cv::Mat format(const cv::Mat& src) = 0;
    };
    
    template<typename R>
    class BaseParser {
    public:
        virtual ~BaseParser() = default;

        virtual int         parse(std::vector<std::shared_ptr<TRT::Tensor>>& output, std::vector<std::shared_ptr<R>>& result, int device=0) const { return 0; }
        virtual const char* get_plugin_name() const { return ""; }
        virtual bool        check_valid(const std::vector<std::shared_ptr<TRT::Tensor>>& outputs) const {
            INFOF("PluginParser Inheritance error!");
            return false;
        }
    };

    template<typename R>
    class OutputParser : virtual public BaseParser<R> {
    public:
        virtual ~OutputParser() = default;
        // device: 0->cpu; 1->gpu
        int parse(std::vector<std::shared_ptr<TRT::Tensor>>& output, std::vector<std::shared_ptr<R>>& result, int device=0) const override {
            TRT::Tensor buffer(TRT::DataType::Float);
            std::vector<int> defect_nums;
            if (device == 0) {
                defect_nums = output2buffer_cpu(output, buffer);
            } else {
                defect_nums = output2buffer_gpu(output, buffer);
            }
        
            buffer2struct(result, buffer, defect_nums);
            return 0;
        }
    protected:
        // 把output tensor中的内容解到统一的buffer tensor中
        // return: 每个图像的目标(缺陷)数
        virtual std::vector<int> output2buffer_cpu(std::vector<std::shared_ptr<TRT::Tensor>>& output, TRT::Tensor& buffer)       const = 0;
        virtual std::vector<int> output2buffer_gpu(const std::vector<std::shared_ptr<TRT::Tensor>>& output, TRT::Tensor& buffer) const = 0;
        // 把buffer中的内容解到Result结构体中
        virtual int buffer2struct(std::vector<std::shared_ptr<R>>& result, TRT::Tensor& buffer, const std::vector<int>& defect_nums) const { return 0; }
    };

    template<typename R>
    class PluginParser : virtual public BaseParser<R> {
    public:
        PluginParser() = default;
        explicit PluginParser(std::string&& r_str) : plugin_name_(r_str) { }
        virtual ~PluginParser() = default;

        virtual const char* get_plugin_name() const override { return plugin_name_.c_str(); }
    protected:
        const std::string plugin_name_;
    };
    
    // 某一任务的推理引擎，该任务以R为结果类型
    template<typename R>
    class Engine {
    public:
        Engine() = default;
        Engine(const std::string& path, const std::shared_ptr<BaseParser<R>> parser) :
            engine_(TRT::load_infer(path)), parser_(parser) {
            if (! parser_) {
                INFOF("parser load fail, please check your parser!");
            }
            if (! engine_) {
                INFOF("Engine load fail, please check the path of plan file!");
            }
            int num_output = engine_->num_output();
            std::vector<std::shared_ptr<TRT::Tensor>> output;
            output.reserve(num_output);
            for (int i = 0; i < num_output; ++i) {
                output.push_back(engine_->output(i));
            }
            FMT_INFO("using %s", parser_->get_plugin_name());
            if (! parser_->check_valid(output)) {
                FMT_INFOW("opt plugin does not match, may cause incorrect result or even CORE!!! please change the parser or re-import onnx file with correct plugin!");
            } else {
                INFO("opt plugin check passed!");
            }

            engine_->print();
        }

        std::shared_ptr<TRT::Infer>& mutable_infer() {
            if (! engine_) {
                INFOF("Engine load fail, please check the path of plan file!");
            }
            return engine_;
        }
        const std::shared_ptr<const TRT::Infer> immutable_infer() {
            if (! engine_) {
                INFOF("Engine load fail, please check the path of plan file!");
            }
            return engine_;
        }

        // 输入预处理后的单张图片，同步执行：塞进input tensor -> forward -> parse output
        std::shared_ptr<R> run(cv::Mat& image, std::array<float, 3>& mean, std::array<float, 3>& std) {
            if (image.empty()) {
                INFOW("Input image is empty, please check input image!");
                return nullptr;
            }
            if (! engine_) {
                INFOF("Engine load fail, please check the path of plan file!");
            }
            
            auto input = engine_->input(0);
            input->set_norm_mat(0, image, mean.data(), std.data());

            int num_output = engine_->num_output();
            std::vector<std::shared_ptr<TRT::Tensor>> output;
            output.reserve(num_output);
            for (int i = 0; i < num_output; ++i) {
                output.push_back(engine_->output(i));
            }

            engine_->forward();
            FMT_INFOD("real defect num: %d", output[0]->at<int>(0, 0));

            std::vector<std::shared_ptr<R>> ret;
            parser_->parse(output, ret);
            if (ret.size() != 1) {
                FMT_INFOW("max batch size of infer(%d) is NOT equal to batch size of input tensor(1), please check your onnx or trt model and do a MODEL COMPILE again! ", engine_->get_max_batch_size());
            }
            return ret[0];
        }

        // TODO: 推理多张图片（batch_size > 1）
        std::vector<std::shared_ptr<R>> run(std::vector<cv::Mat>& images, std::array<float, 3>& mean, std::array<float, 3>& std) {
            if (! engine_) {
                INFOF("Engine load fail, please check the path of plan file!");
            }
            int images_size = images.size();
            int max_batch_size = engine_->get_max_batch_size();
            // 图片数量大于引擎最大批处理数量，assert；TODO:更温和的策略
            if (images_size > max_batch_size) {
                FMT_INFOF("batch images(%d) > infer max_batch_size(%d), please RECOMPILE MODEL or check your input vector!", images_size, max_batch_size);
            } else if (images_size < max_batch_size) {
                FMT_INFOW("batch images(%d) < infer max_batch_size(%d), the performance of engine can be better!", images_size, max_batch_size);
            }

            auto input = engine_->input(0);
            auto image_w = input->width();
            auto image_h = input->height();
            for (int i = 0; i < images.size(); ++i) {
                if (images[i].empty()) {
                    INFOW("index %d in current batch is empty!", i);
                    images[i] = cv::Mat(image_w, image_h, CV_8UC3, cv::Scalar(0, 0, 0));
                }
            }
            int n = 0;
            for (const auto& image : images) {
                input->set_norm_mat(n++, image, mean.data(), std.data());
            }

            int num_output = engine_->num_output();
            std::vector<std::shared_ptr<TRT::Tensor>> output;
            output.reserve(num_output);
            for (int i = 0; i < num_output; ++i) {
                output.push_back(engine_->output(i));
            }

            engine_->forward();

            std::vector<std::shared_ptr<R>> ret;
            ret.reserve(max_batch_size);
            parser_->parse(output, ret);
            if (ret.size() != max_batch_size) {
                INFOW("Unexpected result number!");
            }
            return ret;
        }
    private:
        std::shared_ptr<TRT::Infer> engine_;
        const std::shared_ptr<BaseParser<R>> parser_;
    };
    
    // 创建引擎函数，推理结果类型为R
    template<typename R>
    std::shared_ptr<Engine<R>> create_infer (
        const std::string& path,
        const std::shared_ptr<BaseParser<R>> parser) {
        return std::make_shared<Engine<R>>(path, parser);
    }

    template<typename R>
    class AmirstanPluginParser : public PluginParser<R> {
    public:
        AmirstanPluginParser() : PluginParser<R>("amirstan_plugin") { }
        virtual bool check_valid(const std::vector<std::shared_ptr<TRT::Tensor>>& outputs) const override;
    };

    template<typename R>
    class MMDeployPluginParser : public PluginParser<R> {
    public:
        MMDeployPluginParser() : PluginParser<R>("mmdeploy_plugin") { }
        virtual bool check_valid(const std::vector<std::shared_ptr<TRT::Tensor>>& outputs) const override;
    };
    // and so on ...

}; // namespace App


// 第0维一般是动态batch，不检查
#define CHECK_OUTPUTS_NUM(outputs, outputs_num_expect)  if (outputs.size() != outputs_num_expect) {                         \
                                                            FMT_INFOW(err_info, iLogger::format(                            \
                                                                "output tensor number error, expect num is %d, but got %d", \
                                                                outputs_num_expect, outputs.size()).c_str());               \
                                                            return false;                                                   \
                                                        }
#define CHECK_TENSOR_DATA_TYPE_RET_ASSERT(tensor, output_idx, err_info, type_expect)    if (tensor->type() != type_expect) {                                          \
                                                                                            FMT_INFOW(err_info,                                                       \
                                                                                            iLogger::string_format(                                                   \
                                                                                            "output tensor %d data type error, expect %s, but got %s",                \
                                                                                            output_idx, TRT::data_type_string(type_expect),                           \
                                                                                            TRT::data_type_string(tensor->type())).c_str());                          \
                                                                                            return false;                                                             \
                                                                                        }
#define CHECK_TENSOR_NDIMS_RET_ASSERT(tensor, output_idx, err_info, ndims_expect)   if (tensor->ndims() != ndims_expect) {                         \
                                                                                        FMT_INFOW(err_info,                                        \
                                                                                        iLogger::string_format(                                    \
                                                                                        "output tensor %d ndims error, expect %d, but got %d",     \
                                                                                        output_idx, ndims_expect, tensor->ndims()).c_str());       \
                                                                                        return false;                                              \
                                                                                    }
#define CHECK_2_DIM_TENSOR_SHAPE_RET_ASSERT(tensor, output_idx, err_info, shape_1_expect)   if (tensor->shape(1) != shape_1_expect) {                                        \
                                                                                                FMT_INFOW(err_info,                                                          \
                                                                                                iLogger::string_format(                                                      \
                                                                                                "output tensor %d shape error, expect {batch, %d}, but got {batch, %d}",     \
                                                                                                output_idx, shape_1_expect, tensor->shape(1)).c_str());                      \
                                                                                                return false;                                                                \
                                                                                            }
#define CHECK_3_DIM_TENSOR_SHAPE_RET_ASSERT(tensor, output_idx, err_info, shape_1_expect, shape_2_expect)   if (tensor->shape(1) != shape_1_expect ||                                                \
                                                                                                                tensor->shape(2) != shape_2_expect) {                                                \
                                                                                                                FMT_INFOW(err_info,                                                                  \
                                                                                                                iLogger::string_format(                                                              \
                                                                                                                "output tensor %d shape error, expect {batch, %d, %d}, but got {batch, %d, %d}",     \
                                                                                                                output_idx, shape_1_expect, shape_2_expect,                                          \
                                                                                                                tensor->shape(1), tensor->shape(2)).c_str());                                        \
                                                                                                                return false;                                                                        \
                                                                                                            }
#define CHECK_4_DIM_TENSOR_SHAPE_RET_ASSERT(tensor, output_idx, err_info, shape_1_expect, shape_2_expect, shape_3_expect)   if (tensor->shape(1) != shape_1_expect ||                                                         \
                                                                                                                                tensor->shape(2) != shape_2_expect ||                                                         \
                                                                                                                                tensor->shape(3) != shape_3_expect) {                                                         \
                                                                                                                                FMT_INFOW(err_info,                                                                           \
                                                                                                                                iLogger::string_format(                                                                       \
                                                                                                                                "output tensor %d shape error, expect {batch, %d, %d, %d}, but got {batch, %d, %d, %d}",      \
                                                                                                                                output_idx, shape_1_expect, shape_2_expect, shape_3_expect,                                   \
                                                                                                                                tensor->shape(1), tensor->shape(2), tensor->shape(3)).c_str());                               \
                                                                                                                                return false;                                                                                 \
                                                                                                                            }

template<typename R>
bool App::AmirstanPluginParser<R>::check_valid(const std::vector<std::shared_ptr<TRT::Tensor>>& outputs) const {
    const std::string err_info = "network output format is not matched with amirstan plugin, detail msg: %s";
    CHECK_OUTPUTS_NUM(outputs, 4);
    for (int i = 0; i < 4; ++i) {
        const auto& tensor = outputs[i];
        switch (i) {
            case 0 : {
                // tensor name: num_detections
                // tensor data type: Int32
                // tensor shape:{batch, 1}
                CHECK_TENSOR_DATA_TYPE_RET_ASSERT(tensor, i, err_info, TRT::DataType::Int32);
                CHECK_TENSOR_NDIMS_RET_ASSERT(tensor, i, err_info, 2);
                CHECK_2_DIM_TENSOR_SHAPE_RET_ASSERT(tensor, i, err_info, 1);
                break;
            }
            case 1 : {
                // tensor name: boxes
                // tensor data type: FLoat32
                // tensor shape:{batch, 100, 4}
                CHECK_TENSOR_DATA_TYPE_RET_ASSERT(tensor, i, err_info, TRT::DataType::Float);
                CHECK_TENSOR_NDIMS_RET_ASSERT(tensor, i, err_info, 3);
                CHECK_3_DIM_TENSOR_SHAPE_RET_ASSERT(tensor, i, err_info, 100, 4);
                break;
            }
            case 2 : {
                // tensor name: scores
                // tensor data type: FLoat32
                // tensor shape:{batch, 100}
                CHECK_TENSOR_DATA_TYPE_RET_ASSERT(tensor, i, err_info, TRT::DataType::Float);
                CHECK_TENSOR_NDIMS_RET_ASSERT(tensor, i, err_info, 2);
                CHECK_2_DIM_TENSOR_SHAPE_RET_ASSERT(tensor, i, err_info, 100);
                break;
            }
            case 3 : {
                // tensor name: classes
                // tensor data type: FLoat32
                // tensor shape:{batch, 100}
                CHECK_TENSOR_DATA_TYPE_RET_ASSERT(tensor, i, err_info, TRT::DataType::Float);
                CHECK_TENSOR_NDIMS_RET_ASSERT(tensor, i, err_info, 2);
                CHECK_2_DIM_TENSOR_SHAPE_RET_ASSERT(tensor, i, err_info, 100);
                break;
            }
        }
    }
    return true;
}

template<typename R>
bool App::MMDeployPluginParser<R>::check_valid(const std::vector<std::shared_ptr<TRT::Tensor>>& outputs) const {
    const std::string err_info = "network output format is not matched with mmdeploy tensorrt plugin, detail msg: %s";
    CHECK_OUTPUTS_NUM(outputs, 2);
    for (int i = 0; i < 2; ++i) {
        const auto& tensor = outputs[i];
        switch (i) {
            case 0 : {
                // tensor name: dets
                // tensor data type: FLoat32
                // tensor shape:{batch, 100, 5}
                CHECK_TENSOR_DATA_TYPE_RET_ASSERT(tensor, i, err_info, TRT::DataType::Float);
                CHECK_TENSOR_NDIMS_RET_ASSERT(tensor, i, err_info, 3);
                CHECK_3_DIM_TENSOR_SHAPE_RET_ASSERT(tensor, i, err_info, 100, 5);
                break;
            }
            case 1 : {
                // tensor name: labels
                // tensor data type: Int32
                // tensor shape:{batch, 100}
                CHECK_TENSOR_DATA_TYPE_RET_ASSERT(tensor, i, err_info, TRT::DataType::Int32);
                CHECK_TENSOR_NDIMS_RET_ASSERT(tensor, i, err_info, 2);
                CHECK_2_DIM_TENSOR_SHAPE_RET_ASSERT(tensor, i, err_info, 100);
                break;
            }
        }
    }
    return true;
}

#endif // APP_HPP