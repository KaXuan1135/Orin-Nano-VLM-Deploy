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

    void generate(
        const std::vector<cv::Mat>& images, 
        const std::string& user_prompt,
        const GenerateConfig& gen_config,
        GenerateResult& gen_result
    ) override;

    void extract_visual_features(
        const std::vector<cv::Mat>& images,
        const GenerateConfig& gen_config,
        VisualFeatures& vis_feats
    ) override;

    void generate_from_features(
        const VisualFeatures& visual_features,
        const std::string& user_prompt,
        const GenerateConfig& gen_config,
        GenerateResult& gen_result
    ) override;

    InternVL3VisionEngine vis_engine;
    InternVL3LLMEngine llm_engine;

private:

    ModelConfig m_config;
    cudaStream_t m_stream;

};

} // namespace trt_multimodal