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

SharedGenHandle AsyncInternVL3Runner::enqueue_generate(
    const std::vector<cv::Mat>& images, 
    const std::string& user_prompt,
    const GenerateConfig& gen_config) 
{
    SharedGenHandle dummy;
    return dummy;
    // return llm_engine.generate_from_features(
    //     vis_engine.extract_visual_features(images, gen_config), 
    //     user_prompt, 
    //     gen_config);
}

SharedVisHandle AsyncInternVL3Runner::enqueue_extract_visual_features(
    const std::vector<cv::Mat>& images,
    const GenerateConfig& gen_config
) {

    auto handle = m_sync_runner.vis_engine.enqueue_extract_visual_features(images, gen_config);
    // m_task_map[handle.generate_result.request_id] = handle; // 存入 map 供 listener 查找
    return handle;
}

SharedGenHandle AsyncInternVL3Runner::enqueue_generate_from_features(
    const VisualFeatures& vis_features,
    const std::string& user_prompt,
    const GenerateConfig& gen_config
) {

    auto handle = m_sync_runner.llm_engine.enqueue_generate_from_features(vis_features, user_prompt, gen_config);
    // m_task_map[handle.generate_result.request_id] = handle; // 存入 map 供 listener 查找, different task_map?
    m_inflight_llm_tasks[handle->generate_result.request_id] = handle;
    return handle;

}
void AsyncInternVL3Runner::generate_listener_loop(
) {
    while (!m_stop) {

        std::vector<GenerateResult*> gen_results;
        for (auto& [rid, handle] : m_inflight_llm_tasks) {
            gen_results.push_back(&(handle->generate_result));
        }
        m_sync_runner.llm_engine.update_response(gen_results, 1000, false);
    }
}


} // namespace trt_multimodal