#include "TrtMultimodalRunner/InternVL3/AsyncInternVL3Runner.hpp"

namespace trt_multimodal {

AsyncInternVL3Runner::AsyncInternVL3Runner(const ModelConfig& config) 
    :m_sync_runner(config), m_stop(false)
{
    m_worker = std::thread(&AsyncInternVL3Runner::worker_loop, this);
}

AsyncInternVL3Runner::~AsyncInternVL3Runner() 
{
    m_stop.store(true);
    m_cv.notify_all();
    if (m_worker.joinable()) m_worker.join();
    {
        std::lock_guard<std::mutex> lock(m_map_mutex);
        m_inflight_vis_tasks.clear();
        m_inflight_llm_tasks.clear();
    }

    cudaDeviceSynchronize();
}

SharedVisGenHandle AsyncInternVL3Runner::enqueue_generate(
    const std::vector<cv::Mat>& images, 
    const std::string& user_prompt,
    const GenerateConfig& gen_config) 
{
    SharedVisGenHandle handle = enqueue_extract_visual_features(images, gen_config);
    handle->do_llm.store(true);
    handle->generate_result.gen_config = gen_config;
    handle->generate_result.user_prompt = user_prompt;
    return handle;
}

SharedVisGenHandle AsyncInternVL3Runner::enqueue_extract_visual_features(
    const std::vector<cv::Mat>& images,
    const GenerateConfig& gen_config
) {
    SharedVisGenHandle handle = std::make_shared<VisGenHandle>();
    m_sync_runner.vis_engine.enqueue_extract_visual_features(images, gen_config, handle);
    // runner create the id for vision task, it is not like llm executor has its own request id
    handle->visual_features.request_id = vis_rid++;
    handle->do_vis.store(true);
    m_inflight_vis_tasks[handle->visual_features.request_id] = handle;
    return handle;
}

SharedVisGenHandle AsyncInternVL3Runner::enqueue_generate_from_features(
    const VisualFeatures& vis_features,
    const std::string& user_prompt,
    const GenerateConfig& gen_config
) {
    SharedVisGenHandle handle = std::make_shared<VisGenHandle>();
    m_sync_runner.llm_engine.enqueue_generate_from_features(vis_features, user_prompt, gen_config, handle);
    handle->do_llm.store(true);
    m_inflight_llm_tasks[handle->generate_result.request_id] = handle;
    return handle;
}

void AsyncInternVL3Runner::worker_loop(
) {
    while (!m_stop) {
        std::vector<GenerateResult*> gen_results;
        {
            std::lock_guard<std::mutex> lock(m_map_mutex);
            if (m_inflight_vis_tasks.empty() && m_inflight_llm_tasks.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // See which vision has done, and if it need to auto do llm
            for (auto it = m_inflight_vis_tasks.begin(); it != m_inflight_vis_tasks.end(); ) {
                auto& handle = it->second;
                
                if (handle->vis_finished.load()) {
                    if (handle->do_llm.load()) {
                        std::cout << "Task " << handle->visual_features.request_id << " moved to LLM" << std::endl;
                        m_sync_runner.llm_engine.enqueue_generate_from_features(handle->visual_features, handle->generate_result.user_prompt, handle->generate_result.gen_config, handle);
                        m_inflight_llm_tasks[handle->generate_result.request_id] = handle;
                    }
                    it = m_inflight_vis_tasks.erase(it);
                } else {
                    ++it;
                }
            }

            for (auto& [rid, handle] : m_inflight_llm_tasks) {
                gen_results.push_back(&(handle->generate_result));
            }
        }

        std::cout << gen_results.size() << " gen running" << std::endl;
        m_sync_runner.llm_engine.update_response(gen_results, 1000, false);

        for (size_t i = 0; i < gen_results.size(); ++i) {
            auto& res = gen_results[i];
            if (!res->outputs_tokens.empty()) {
                std::cout << "Req [" << res->request_id << "] generating... tokens: " 
                        << res->outputs_tokens[0].size() << std::endl;
            }
        }

        {
            std::lock_guard<std::mutex> lock(m_map_mutex);

            for (auto it = m_inflight_llm_tasks.begin(); it != m_inflight_llm_tasks.end(); ) {
                auto& handle = it->second;
                
                if (handle->generate_result.done_output.load()) {
                    handle->gen_finished.store(true);
                    it = m_inflight_llm_tasks.erase(it);
                } else {
                    ++it;
                }
            }
        }

        
    }
}

} // namespace trt_multimodal