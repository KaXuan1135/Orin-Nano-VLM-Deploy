#include "TrtMultimodalRunner/InternVL3/AsyncInternVL3Runner.hpp"

namespace trt_multimodal {

AsyncInternVL3Runner::AsyncInternVL3Runner(const ModelConfig& config) 
    :m_sync_runner(config), m_stop(false)
{
    m_worker = std::thread(&AsyncInternVL3Runner::generate_listener_loop, this);
}

AsyncInternVL3Runner::~AsyncInternVL3Runner() 
{
    m_stop = true;
    m_cv.notify_all();
    if (m_worker.joinable()) m_worker.join();
    // delete m_sync_runner?
}

SharedGenHandle AsyncInternVL3Runner::enqueue_generate(
    const std::vector<cv::Mat>& images, 
    const std::string& user_prompt,
    const GenerateConfig& gen_config) 
{
    auto gen_handle = SharedGenHandle();

    // TODO

    return gen_handle;
}

SharedVisHandle AsyncInternVL3Runner::enqueue_extract_visual_features(
    const std::vector<cv::Mat>& images,
    const GenerateConfig& gen_config
) {
    auto handle = m_sync_runner.vis_engine.enqueue_extract_visual_features(images, gen_config);
    // runner create the id for vision task, it is not like llm executor has its own request id
    handle->visual_features.request_id = vis_rid++;
    m_inflight_vis_tasks[handle->visual_features.request_id] = handle;
    return handle;
}

SharedGenHandle AsyncInternVL3Runner::enqueue_generate_from_features(
    const VisualFeatures& vis_features,
    const std::string& user_prompt,
    const GenerateConfig& gen_config
) {
    auto handle = m_sync_runner.llm_engine.enqueue_generate_from_features(vis_features, user_prompt, gen_config);
    m_inflight_llm_tasks[handle->generate_result.request_id] = handle;
    return handle;
}

void AsyncInternVL3Runner::generate_listener_loop(
) {
    while (!m_stop) {

        std::vector<GenerateResult*> gen_results;

        {
            std::lock_guard<std::mutex> lock(m_map_mutex);
            if (m_inflight_llm_tasks.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            for (auto& [rid, handle] : m_inflight_llm_tasks) {
                gen_results.push_back(&(handle->generate_result));
            }
        }

        m_sync_runner.llm_engine.update_response(gen_results, 1000, false);
    }
}


} // namespace trt_multimodal