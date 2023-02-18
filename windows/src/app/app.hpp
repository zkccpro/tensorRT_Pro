

#ifndef APP_HPP
#define APP_HPP

#include <string>
#include <vector>
#include <array>
#include <memory>

#include <plugin/amirInferPlugin.h>
#include <infer/trt_infer.hpp>
#include <opencv2/opencv.hpp>
#include <ilogger.hpp>

namespace App {
    class Result {
    public:
        virtual ~Result() = default;
        virtual std::string format() = 0;
        virtual cv::Mat format(const cv::Mat& src) = 0;
    };
    
    template<typename R>
    class OutputParser {
    public:
        virtual ~OutputParser() = default;
        // device: 0->cpu; 1->gpu
        int parse(std::vector<std::shared_ptr<TRT::Tensor>>& output, std::vector<std::shared_ptr<R>>& result, int device=0) {
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
        virtual int buffer2struct(std::vector<std::shared_ptr<R>>& result, TRT::Tensor& buffer, const std::vector<int>& defect_nums) { return 0; }
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
            engine_->print();
        }

        std::shared_ptr<TRT::Infer> mutable_infer() {return engine_;}

        // 输入预处理后的单张图片，同步执行：塞进input tensor -> forward -> parse output
        std::shared_ptr<R> run(cv::Mat& image, std::array<float, 3>& mean, std::array<float, 3>& std) {
            if (image.empty()) {
                INFOW("Input image is empty, please check input image!");
                return nullptr;
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
            ret.reserve(images_size);
            parser_->parse(output, ret);
            if (ret.size() != max_batch_size) {
                INFOW("Unexpected result number!");
            }
            return ret;
        }
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