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

void InternVL3Runner::generate(
    const std::vector<cv::Mat>& images, 
    const std::string& user_prompt,
    const GenerateConfig& gen_config,
    GenerateResult& gen_result
) {
    VisualFeatures vis_feats = VisualFeatures();
    vis_engine.extract_visual_features(images, gen_config, vis_feats, true);
    
    llm_engine.generate_from_features(
        vis_feats, 
        user_prompt, 
        gen_config,
        gen_result
    );
}

void InternVL3Runner::extract_visual_features(
    const std::vector<cv::Mat>& images,
    const GenerateConfig& gen_config,
    VisualFeatures& vis_feats
) {
    vis_engine.extract_visual_features(images, gen_config, vis_feats, true);
}

void InternVL3Runner::generate_from_features(
    const VisualFeatures& vis_features,
    const std::string& user_prompt,
    const GenerateConfig& gen_config,
    GenerateResult& gen_result
) {
    llm_engine.generate_from_features(vis_features, user_prompt, gen_config, gen_result);
}

} // namespace trt_multimodal