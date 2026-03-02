#pragma once
#include <vector>
#include <string>
#include <memory>

#include "Types.hpp"

namespace cv { class Mat; }

namespace trt_multimodal {

class IMultimodalRunner {
public:
    virtual ~IMultimodalRunner() = default;

    static std::unique_ptr<IMultimodalRunner> create(
        const ModelConfig& model_config
    );

    static std::unique_ptr<IMultimodalRunner> initialize();

    virtual GenerateResult generate(
        const std::vector<cv::Mat>& images, 
        const std::string& user_prompt,
        const GenerateConfig& gen_config
    ) = 0; 

    virtual VisualFeatures extract_visual_features(
        const std::vector<cv::Mat>& images,
        const GenerateConfig& gen_config
    ) = 0;

    virtual GenerateResult generate_from_features(
        const VisualFeatures& visual_features,
        const std::string& user_prompt,
        const GenerateConfig& gen_config
    ) = 0;

};

} // namespace trt_multimodal