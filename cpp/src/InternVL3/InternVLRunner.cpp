#include "TrtMultimodalRunner/InternVL3/InternVL3Runner.hpp"

namespace trt_multimodal {

InternVL3Runner::InternVL3Runner(
    const ModelConfig& config
) : m_config(config){
    cudaError_t status = cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) throw std::runtime_error("Failed to create CUDA stream");

    vis_engine = std::make_unique<InternVL3VisionEngine>(m_config, m_stream);
    llm_engine = std::make_unique<InternVL3LLMEngine>(m_config, m_stream);
}

InternVL3Runner::~InternVL3Runner() {
    if (m_stream) cudaStreamDestroy(m_stream);
}

void InternVL3Runner::generate(
    const std::vector<std::vector<cv::Mat>>& images, 
    const std::vector<std::string>& user_prompt,
    const std::vector<GenerateConfig>& gen_config,
    std::vector<GenerateResult>& gen_result
) {

    size_t batch_size = images.size();
    std::vector<SharedVisGenHandle> handles(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
        handles[b] = std::make_shared<VisGenHandle>();
        handles[b]->generate_result.user_prompt = user_prompt[b];
        handles[b]->gen_config = gen_config[b];
    }

    std::vector<VisualFeatures> vis_feats(images.size());
    for (int i = 0; i < images.size(); ++i) {
        vis_engine->extract_visual_features(images[i], gen_config[i], handles[i]->visual_features, true);
    }

    llm_engine->generate_from_features(handles);

    gen_result.clear();
    gen_result.reserve(handles.size());

    for (const auto& handle : handles) {
        gen_result.push_back(handle->generate_result);
    }

}

} // namespace trt_multimodal