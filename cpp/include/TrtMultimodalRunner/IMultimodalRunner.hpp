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

    static SharedVisGenHandle create_handle(
        const GenerateConfig& gen_config,
        const std::string& user_prompt,
        const std::vector<cv::Mat>& images = {}, 
        const std::vector<SharedVisGenHandle>& history_handles = {}
    ) {
        SharedVisGenHandle handle = std::make_shared<VisGenHandle>();
        handle->gen_config = gen_config;
        handle->history_handles = history_handles;
        handle->generate_result.user_prompt = user_prompt;
        handle->visual_features.images = images;

        return handle;
    }

    virtual void generate(
        const std::vector<std::vector<cv::Mat>>& images, 
        const std::vector<std::string>& user_prompt,
        const std::vector<GenerateConfig>& gen_config,
        std::vector<GenerateResult>& gen_result
    ) = 0; 

};

} // namespace trt_multimodal