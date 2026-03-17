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
        const std::vector<std::vector<cv::Mat>>& images, 
        const std::vector<std::string>& user_prompt,
        const std::vector<GenerateConfig>& gen_config,
        std::vector<GenerateResult>& gen_result
    ) override;

    std::unique_ptr<InternVL3VisionEngine> vis_engine;
    std::unique_ptr<InternVL3LLMEngine> llm_engine;

private:

    ModelConfig m_config;
    cudaStream_t m_stream;

};

} // namespace trt_multimodal