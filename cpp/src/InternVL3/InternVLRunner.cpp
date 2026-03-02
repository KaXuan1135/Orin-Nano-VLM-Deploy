#include "TrtMultimodalRunner/InternVL3/InternVL3Runner.hpp"

namespace trt_multimodal {

InternVL3Runner::InternVL3Runner(const ModelConfig& config) : m_config(config){
    cudaError_t status = cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) throw std::runtime_error("Failed to create CUDA stream");

    vis_engine = InternVL3VisionEngine();
    llm_engine = InternVL3LLMEngine();

    vis_engine.init(config, m_stream);
    llm_engine.init(config, m_stream);
}

InternVL3Runner::~InternVL3Runner() {
    if (m_stream) cudaStreamDestroy(m_stream);
}

GenerateResult InternVL3Runner::generate(
    const std::vector<cv::Mat>& images, 
    const std::string& user_prompt,
    const GenerateConfig& gen_config) 
{
    return llm_engine.generate_from_features(
        vis_engine.extract_visual_features(images, gen_config), 
        user_prompt, 
        gen_config);
}

VisualFeatures InternVL3Runner::extract_visual_features(
    const std::vector<cv::Mat>& images,
    const GenerateConfig& gen_config
) {
    return vis_engine.extract_visual_features(images, gen_config);
}

GenerateResult InternVL3Runner::generate_from_features(
    const VisualFeatures& vis_features,
    const std::string& user_prompt,
    const GenerateConfig& gen_config
) {
    return llm_engine.generate_from_features(vis_features, user_prompt, gen_config);
}

} // namespace trt_multimodal