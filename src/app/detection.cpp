
#include "detection.h"

namespace Detection {
    DetResult::DetResult(const std::vector<BBox>& bboxes) : bboxes_(bboxes) { }

    std::string DetResult::format() {
        std::string ret_str = "result is:\n" ;
        if (bboxes_.size() == 0) {
            ret_str.append("empty.");
        }
        for (int i = 0; i < defect_num(); ++i) {
            ret_str.append(iLogger::string_format("obj%d: left=%f, top=%f, right=%f, bottom=%f, confidence=%f, label=%f\n",
                                        i + 1, bboxes_[i].left, bboxes_[i].top, bboxes_[i].right, bboxes_[i].bottom, bboxes_[i].confidence, bboxes_[i].label));
        }
        return ret_str;
    }

    cv::Mat DetResult::format(const cv::Mat& src) {
        if (src.empty()) {
            INFOW("Format input image is empty, may cause core when you try to save the image!");
        }
        cv::Mat dst { src };
        for (const auto& bbox : bboxes_) {
            if ((bbox.right - bbox.left) != 0 &&
                (bbox.bottom - bbox.top) != 0) {
                auto rect = cv::Rect(bbox.left, bbox.top, bbox.right - bbox.left, bbox.bottom - bbox.top);
                cv::rectangle(dst, rect, cv::Scalar(0x27, 0xC1, 0x36), 2);
                // TODO: class text
                cv::putText(dst, std::to_string(bbox.label), cv::Point(rect.x, rect.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
        }
        return dst;
    }

    int DetectionParser::buffer2struct(std::vector<std::shared_ptr<DetResult>>& result, TRT::Tensor& buffer, const std::vector<int>& defect_nums) const {
        // if buffer is on gpu: buffer.to_cpu();
        int batch_size = buffer.shape(0);
        // if batch_size != defect_nums.size(): 报错
        // if batch_size > 0: 报错
        result.reserve(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            std::vector<BBox> bboxes;
            for (int j = 0; j < defect_nums[i] * NUM_BBOX_ELEMENT; j += NUM_BBOX_ELEMENT) {
                bboxes.emplace_back(buffer.at<float>(i, j), buffer.at<float>(i, j + 1),
                                        buffer.at<float>(i, j + 2), buffer.at<float>(i, j + 3), 
                                        buffer.at<float>(i, j + 4), buffer.at<int>(i, j + 5));
            }
            result.emplace_back(std::make_shared<DetResult>(bboxes));
        }
        return 0;
    }

    std::vector<int> AmirstanDetectionParser::output2buffer_cpu(std::vector<std::shared_ptr<TRT::Tensor>>& output, TRT::Tensor& buffer) const {
        auto& defects_info = output[0]; // shape: batch*1
        auto& bboxes_info = output[1]->to_cpu(); // shape: batch*100*4
        auto& scores_info = output[2]->to_cpu(); // shape: batch*100
        auto& classes_info = output[3]->to_cpu(); // shape: batch*100
        std::vector<int> defect_nums;
        int batch_size = defects_info->shape(0);
        FMT_INFOD("parse batch_size: %d", batch_size);
        buffer.resize(batch_size, MAX_IMAGE_BBOX * NUM_BBOX_ELEMENT).to_cpu();
        for (int i = 0; i < batch_size; ++i) { // batch
            int defect_num = defects_info->at<int>(i, 0);
            FMT_INFOD("parse defect_num: %d", defect_num);
            defect_nums.push_back(defect_num);
            for (int j = 0; j < defect_num; ++j) { // defect
                for (int k = 0; k < 4; ++k) { // left, top, right, bottom
                    buffer.at<float>(i, j * NUM_BBOX_ELEMENT + k) = bboxes_info.at<float>(i, j, k);
                }
                buffer.at<float>(i, j * NUM_BBOX_ELEMENT + 4) = scores_info.at<float>(i, j);
                buffer.at<float>(i, j * NUM_BBOX_ELEMENT + 5) = classes_info.at<float>(i, j);
            }
        }
        return defect_nums;
    }

    std::vector<int> MMDeployDetectionParser::output2buffer_cpu(std::vector<std::shared_ptr<TRT::Tensor>>& output, TRT::Tensor& buffer) const {
        auto& defs_info = output[0]->to_cpu(); // shape: batch*100*5, (left, top, right, bottom, score)
        auto& labels_info = output[1]->to_cpu(); // shape: batch*100
        std::vector<int> defect_nums;
        int batch_size = defs_info.shape(0);
        FMT_INFOD("parse batch_size: %d", batch_size);
        buffer.resize(batch_size, MAX_IMAGE_BBOX * NUM_BBOX_ELEMENT).to_cpu();
        for (int i = 0; i < batch_size; ++i) { // batch
        int defect_num { 0 };
            for (int z = 0; z < MAX_IMAGE_BBOX; ++z) {
                if (defs_info.at<float>(i, z, 4) > 0.) {
                    ++defect_num;
                } else {
                    break;
                }
            }
            FMT_INFOD("parse defect_num: %d", defect_num);
            defect_nums.push_back(defect_num);
            for (int j = 0; j < defect_num; ++j) { // defect
                for (int k = 0; k < 4; ++k) { // left, top, right, bottom
                    buffer.at<float>(i, j * NUM_BBOX_ELEMENT + k) = defs_info.at<float>(i, j, k);
                }
                buffer.at<float>(i, j * NUM_BBOX_ELEMENT + 4) = defs_info.at<float>(i, j, 4);
                buffer.at<int>(i, j * NUM_BBOX_ELEMENT + 5) = labels_info.at<int>(i, j);
            }
        }
        return defect_nums;
    }
    
}; // namespace Detection