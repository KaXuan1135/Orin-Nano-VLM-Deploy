#pragma once
#include <iostream>
#include <NvInfer.h>

#include "TrtMultimodalRunner/Types.hpp" 

namespace trt_multimodal {

class InternVL3VisionEngine {
public:

    int init(
        const ModelConfig& config,
        const cudaStream_t& stream
    );    

    void extract_visual_features(
        const std::vector<cv::Mat>& images,
        const GenerateConfig& gen_config,
        VisualFeatures& vis_feats
    );

    SharedVisHandle enqueue_extract_visual_features(
        const std::vector<cv::Mat>& images,
        const GenerateConfig& gen_config
    );

    
private:

    struct VisionSession {
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;

        ~VisionSession() {
            if (context) delete context;
            if (engine)  delete engine;
            if (runtime) delete runtime;
        }
    };

    std::unique_ptr<VisionSession> vis_engine;
    ModelConfig m_config;
    cudaStream_t m_stream;

    class TRTLogger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kINFO) {
                std::string label;
                switch (severity) {
                    case Severity::kINTERNAL_ERROR: label = "[FATAL]";   break;
                    case Severity::kERROR:          label = "[ERROR]";   break;
                    case Severity::kWARNING:        label = "[WARNING]"; break;
                    case Severity::kINFO:           label = "[INFO]";    break;
                    case Severity::kVERBOSE:        label = "[VERBOSE]"; break;
                    default:                        label = "[UNKNOWN]"; break;
                }
                std::cout << "[TensorRT-VIS]" << label << " " << msg << std::endl;
            }
        }
    } gLogger;
    
    AspectRatio find_closest_aspect_ratio(float aspect_ratio, int min_num, int max_num, int image_size);
    std::vector<cv::Mat> dynamic_preprocess(const cv::Mat& image, int min_num, int max_num, int image_size, bool use_thumbnail);
};

} // namespace trt_multimodal