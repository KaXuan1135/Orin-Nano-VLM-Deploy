#include "tensorrt_llm/executor/executor.h"

#include "TrtMultimodalRunner/InternVL3/InternVL3Runner.hpp"
#include "TrtMultimodalRunner/InternVL3/InternVL3LLMEngine.hpp"
#include "TrtMultimodalRunner/InternVL3/InternVL3VisionEngine.hpp"

namespace trt_multimodal {

class InternVL3Runner::Impl {

public:
    explicit Impl(const ModelConfig& config) : m_config(config) {

        cudaError_t status = cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
        if (status != cudaSuccess) throw std::runtime_error("Failed to create CUDA stream");

        vis_engine = InternVL3VisionEngine();
        llm_engine = InternVL3LLMEngine();

        vis_engine.init(config, m_stream);
        llm_engine.init(config, m_stream);
        
    }

    ~Impl() {
        if (m_stream) {
            cudaStreamDestroy(m_stream);
        }
    }

    GenerateResult generate(
        const std::vector<cv::Mat>& images, 
        const std::string& user_prompt,
        const GenerateConfig& gen_config) 
    {
        return llm_engine.generate_from_features(
            vis_engine.extract_visual_features(images, gen_config), 
            user_prompt, 
            gen_config);
    }

    VisualFeatures extract_visual_features(
        const std::vector<cv::Mat>& images,
        const GenerateConfig& gen_config
    ) {
        return vis_engine.extract_visual_features(images, gen_config);
    }

    GenerateResult generate_from_features(
        const VisualFeatures& vis_features,
        const std::string& user_prompt,
        const GenerateConfig& gen_config
    ) {
        return llm_engine.generate_from_features(vis_features, user_prompt, gen_config);
    }

private:

    InternVL3VisionEngine vis_engine;
    InternVL3LLMEngine llm_engine;

    ModelConfig m_config;
    cudaStream_t m_stream;
    // std::unique_ptr<Tokenizer> m_tokenizer;

    double get_duration_ms(std::chrono::steady_clock::time_point start) {
        auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

// --- 外部类代理实现 ---
InternVL3Runner::InternVL3Runner(const ModelConfig& config) 
    : pimpl(std::make_unique<Impl>(config)) {}

InternVL3Runner::~InternVL3Runner() = default;

GenerateResult InternVL3Runner::generate(
    const std::vector<cv::Mat>& images, 
    const std::string& user_prompt,
    const GenerateConfig& gen_config) 
{
    return pimpl->generate(images, user_prompt, gen_config);
}

VisualFeatures InternVL3Runner::extract_visual_features(
    const std::vector<cv::Mat>& images,
    const GenerateConfig& gen_config
) {
    return pimpl->extract_visual_features(images, gen_config);
}

GenerateResult InternVL3Runner::generate_from_features(
    const VisualFeatures& vis_features,
    const std::string& user_prompt,
    const GenerateConfig& gen_config
) {
    return pimpl->generate_from_features(vis_features, user_prompt, gen_config);
}

} // namespace trt_multimodal