#pragma once
#include "TrtMultimodalRunner/IMultimodalRunner.hpp"
#include "TrtMultimodalRunner/Types.hpp" 

namespace trt_multimodal {

class InternVL3Runner : public IMultimodalRunner {
public:

    // User can only create this class by calling IMultimodalRunner create function
    explicit InternVL3Runner(const ModelConfig& config); 

    ~InternVL3Runner() override;

    GenerateResult generate(
        const std::vector<cv::Mat>& images, 
        const std::string& user_prompt,
        const GenerateConfig& gen_config
    ) override;

    VisualFeatures extract_visual_features(
        const std::vector<cv::Mat>& images,
        const GenerateConfig& gen_config
    ) override;

    GenerateResult generate_from_features(
        const VisualFeatures& visual_features,
        const std::string& user_prompt,
        const GenerateConfig& gen_config
    ) override;

private:

    class Impl; 
    std::unique_ptr<Impl> pimpl;

};

} // namespace trt_multimodal