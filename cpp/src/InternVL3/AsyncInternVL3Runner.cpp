#include "TrtMultimodalRunner/InternVL3/AsyncInternVL3Runner.hpp"

namespace trt_multimodal {

AsyncInternVL3Runner::AsyncInternVL3Runner(const ModelConfig& config) 
    :m_sync_runner(config), m_stop(false)
{
    // m_worker = std::thread(&AsyncInternVL3Runner::worker_loop, this);
}

AsyncInternVL3Runner::~AsyncInternVL3Runner() 
{
    m_stop = true;
    m_cv.notify_all();
    if (m_worker.joinable()) m_worker.join();
    // delete m_sync_runner?
}

void AsyncInternVL3Runner::generate_async(
    const std::vector<cv::Mat>& images, 
    const std::string& user_prompt,
    const GenerateConfig& gen_config) 
{
    // return llm_engine.generate_from_features(
    //     vis_engine.extract_visual_features(images, gen_config), 
    //     user_prompt, 
    //     gen_config);
}

void AsyncInternVL3Runner::extract_visual_features_async(
    const std::vector<cv::Mat>& images,
    const GenerateConfig& gen_config
) {
    // return vis_engine.extract_visual_features(images, gen_config);
}

void AsyncInternVL3Runner::generate_from_features_async(
    const VisualFeatures& vis_features,
    const std::string& user_prompt,
    const GenerateConfig& gen_config
) {
    // return llm_engine.generate_from_features(vis_features, user_prompt, gen_config);
}

} // namespace trt_multimodal