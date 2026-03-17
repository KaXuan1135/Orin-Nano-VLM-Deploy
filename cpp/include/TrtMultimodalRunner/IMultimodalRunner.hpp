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

    virtual void generate(
        const std::vector<std::vector<cv::Mat>>& images, 
        const std::vector<std::string>& user_prompt,
        const std::vector<GenerateConfig>& gen_config,
        std::vector<GenerateResult>& gen_result
    ) = 0; 

};

} // namespace trt_multimodal