#pragma once

#include "TrtMultimodalRunner/Types.hpp" 
#include "TrtMultimodalRunner/IMultimodalRunner.hpp"
#include "TrtMultimodalRunner/InternVL3/InternVL3LLMEngine.hpp"
#include "TrtMultimodalRunner/InternVL3/InternVL3VisionEngine.hpp"

namespace trt_multimodal {

class InternVL3Runner : public IMultimodalRunner {
public:

    // User should only create this class by calling IMultimodalRunner create function
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

    InternVL3VisionEngine vis_engine;
    InternVL3LLMEngine llm_engine;

    ModelConfig m_config;
    cudaStream_t m_stream;

};

} // namespace trt_multimodal