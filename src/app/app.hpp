

#ifndef APP_HPP
#define APP_HPP

#include <string>
#include <vector>
#include <plugin/amirInferPlugin.h>
#include <infer/trt_infer.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <ilogger.hpp>

namespace App {
    class Result {
    public:
        virtual std::string format() = 0;
        virtual cv::Mat format(const cv::Mat& src) = 0;
    };
    
    template<typename R>
    class OutputParser {
    public:
        // device: 0->cpu; 1->gpu
        int parse(std::vector<std::shared_ptr<TRT::Tensor>>& output, std::vector<R>& result, int device=0) {
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
        virtual std::vector<int> output2buffer_cpu(std::vector<std::shared_ptr<TRT::Tensor>>& output, TRT::Tensor& buffer) = 0;
        virtual std::vector<int> output2buffer_gpu(const std::vector<std::shared_ptr<TRT::Tensor>>& output, TRT::Tensor& buffer) = 0;
        // 把buffer中的内容解到Result结构体中
        virtual int buffer2struct(std::vector<R>& result, TRT::Tensor& buffer, const std::vector<int>& defect_nums) { return 0; }
    };
    
    // 某一任务的推理引擎，该任务以R为结果类型
    template<typename R>
    class Engine {
    public:
        Engine() = default;
        Engine(const std::string& path, OutputParser<R>& parser) :
            engine_(TRT::load_infer(path)), parser_(&parser) {
            if (! engine_) {
                INFOF("Engine is nullptr, please check the path of infer!");
            }
        }
        // 输入预处理后的单张图片，同步执行：塞进input tensor -> forward -> parse output
        R run(cv::Mat& image) {
            if (image.empty()) {
                INFOF("Input image is empty, please check input image!");
            }
            auto input = engine_->input(0);
            int num_output = engine_->num_output();
            std::vector<std::shared_ptr<TRT::Tensor>> output;
            output.reserve(num_output);
            for (int i = 0; i < num_output; ++i) {
                output.push_back(engine_->output(i));
            }
            
            float mean[] = {0, 0, 0};
            float std[]  = {1, 1, 1};
            input->set_norm_mat(0, image, mean, std);

            engine_->forward();
            INFOD(iLogger::string_format("real defect num: %d", output[0]->at<int>(0, 0)).c_str());
            std::vector<R> ret;
            parser_->parse(output, ret);
            // if ret.size() != 1: 报错
            return ret[0];
        }
        // TODO: 推理多张图片（batch_size > 1）
        std::vector<R> run(std::vector<cv::Mat> images) { }
    private:
        std::shared_ptr<TRT::Infer> engine_;
        OutputParser<R>* parser_;
    };
    
    // 创建引擎函数，推理结果类型为R
    template<typename R>
    std::shared_ptr<Engine<R>> create_infer (
        const std::string& path,
        OutputParser<R>& parser) {
        return std::make_shared<Engine<R>>(path, parser);
    }

    // 启用amirstan_plugin算子
    int use_amirstan_plugin();
    
}; // namespace App


#endif // APP_HPP