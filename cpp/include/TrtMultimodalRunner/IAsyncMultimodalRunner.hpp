#pragma once
#include <vector>
#include <string>
#include <memory>

#include "Types.hpp"

namespace cv { class Mat; }

namespace trt_multimodal {

class IAsyncMultimodalRunner {
public:
    virtual ~IAsyncMultimodalRunner() = default;

    static std::unique_ptr<IAsyncMultimodalRunner> create(
        const ModelConfig& model_config
    );

    static std::unique_ptr<IAsyncMultimodalRunner> initialize();

    virtual void generate_async(
        const std::vector<cv::Mat>& images, 
        const std::string& user_prompt,
        const GenerateConfig& gen_config
    ) = 0; 

    virtual void extract_visual_features_async(
        const std::vector<cv::Mat>& images,
        const GenerateConfig& gen_config
    ) = 0;

    virtual void generate_from_features_async(
        const VisualFeatures& visual_features,
        const std::string& user_prompt,
        const GenerateConfig& gen_config
    ) = 0;

};

} // namespace trt_multimodal