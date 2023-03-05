

#ifndef detection_H
#define detection_H

#include "app.hpp"

namespace Detection {

    const int NUM_BBOX_ELEMENT = 6;
    const int MAX_IMAGE_BBOX = 100;

    // 检测任务的边界框
    struct BBox {
        BBox() = default;
        BBox(float left, float top, float right, float bottom, 
             float confidence, float label):
        left(left), top(top), right(right), bottom(bottom),
        confidence(confidence), label(label),
        area((right - left) * (bottom - top)) { }

        float left, top, right, bottom, confidence, label, area;
    };
    
    // 检测任务的结果
    // TODO: 加上confidence_threshold
    class DetResult : public App::Result {
    public:
        DetResult() = default;
        DetResult(const std::vector<BBox>& bboxes);
        
        virtual std::string format() override;
        virtual cv::Mat     format(const cv::Mat& src) override;

        std::vector<BBox>&  mutable_defects()         { return bboxes_; }
        std::vector<BBox>&  mutable_objects()         { return bboxes_; }
        std::vector<BBox>&  mutable_bboxes()          { return bboxes_; }

        std::vector<BBox>   immutable_defects() const { return bboxes_; }
        std::vector<BBox>   immutable_objects() const { return bboxes_; }
        std::vector<BBox>   immutable_bboxes()  const { return bboxes_; }

        uint32_t defect_num() { return bboxes_.size(); }
        bool     ok()         { return defect_num() == 0; } // 没有缺陷，良品
        bool     negative()   { return defect_num() == 0; } // 没有目标
    private:
        std::vector<BBox> bboxes_;
    };
    
    /// 各个算子库的 output parser
    class DetectionParser : public App::OutputParser<DetResult> {
    protected:
        virtual std::vector<int> output2buffer_cpu(std::vector<std::shared_ptr<TRT::Tensor>>& output, TRT::Tensor& buffer)       const = 0;
        virtual std::vector<int> output2buffer_gpu(const std::vector<std::shared_ptr<TRT::Tensor>>& output, TRT::Tensor& buffer) const = 0;
        virtual int              buffer2struct(std::vector<std::shared_ptr<DetResult>>& result, TRT::Tensor& buffer, const std::vector<int>& defect_nums) const;
    };

    class AmirstanDetectionParser : public DetectionParser, public App::AmirstanPluginParser<DetResult> {
    protected:
        virtual std::vector<int> output2buffer_cpu(std::vector<std::shared_ptr<TRT::Tensor>>& output, TRT::Tensor& buffer)       const override;
        virtual std::vector<int> output2buffer_gpu(const std::vector<std::shared_ptr<TRT::Tensor>>& output, TRT::Tensor& buffer) const override { return {}; }
    };

    class MMDeployDetectionParser : public DetectionParser, public App::MMDeployPluginParser<DetResult> {
    protected:
        virtual std::vector<int> output2buffer_cpu(std::vector<std::shared_ptr<TRT::Tensor>>& output, TRT::Tensor& buffer)       const override;
        virtual std::vector<int> output2buffer_gpu(const std::vector<std::shared_ptr<TRT::Tensor>>& output, TRT::Tensor& buffer) const override { return {}; }
    };
    // and so on ...

    /// 各个parser的全局实例化，todo: 改成全局注册
    const static std::shared_ptr<AmirstanDetectionParser> amirstan_det_plg_parser {std::make_shared<AmirstanDetectionParser>()};
    const static std::shared_ptr<MMDeployDetectionParser> mmdeploy_det_plg_parser {std::make_shared<MMDeployDetectionParser>()};
    // and so on ...
    
}; // namespace Detection


#endif // detection_H