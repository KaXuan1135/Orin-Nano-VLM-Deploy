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

SharedVisGenHandle AsyncInternVL3Runner::enqueue_chat(
    const std::vector<cv::Mat>& images, 
    const std::string& user_prompt,
    const GenerateConfig& gen_config,
    const std::vector<SharedVisGenHandle>& prev_handles) 
{
    SharedVisGenHandle handle = enqueue_extract_visual_features(images, gen_config);
    handle->llm_task_id = llm_rid++;
    handle->do_llm.store(true);
    handle->generate_result.gen_config = gen_config;
    handle->generate_result.user_prompt = user_prompt;
    return handle;
}

SharedVisGenHandle AsyncInternVL3Runner::enqueue_generate(
    const std::vector<cv::Mat>& images, 
    const std::string& user_prompt,
    const GenerateConfig& gen_config,
    const std::vector<SharedVisGenHandle>& prev_handles) // temp solution
{
    SharedVisGenHandle handle = enqueue_extract_visual_features(images, gen_config);
    handle->llm_task_id = llm_rid++;
    handle->do_llm.store(true);
    handle->generate_result.gen_config = gen_config;
    handle->generate_result.user_prompt = user_prompt;
    handle->prev_handles = prev_handles;
    return handle;
}

SharedVisGenHandle AsyncInternVL3Runner::enqueue_extract_visual_features(
    const std::vector<cv::Mat>& images,
    const GenerateConfig& gen_config
) {
    SharedVisGenHandle handle = std::make_shared<VisGenHandle>();
    handle->visual_features.images = images;
    handle->visual_features.gen_config = gen_config;
    handle->vis_task_id = vis_rid++;
    handle->do_vis.store(true);
    {
        std::lock_guard<std::mutex> lock(m_vis_queue_mutex);
        m_queue_vis_tasks.push_back(handle);
        std::cout << "[VIS Task] : Queue " << m_queue_vis_tasks.size() << std::endl;
    }
    return handle;
}

void AsyncInternVL3Runner::enqueue_generate_from_features(
    SharedVisGenHandle& handle,
    // const VisualFeatures& vis_features,
    const std::string& user_prompt,
    const GenerateConfig& gen_config
) {
    handle->llm_task_id = llm_rid++;
    handle->do_llm.store(true);
    handle->generate_result.gen_config = gen_config;
    handle->generate_result.user_prompt = user_prompt;
    {
        std::lock_guard<std::mutex> lock(m_llm_queue_mutex);
        m_queue_llm_tasks.push_back(handle);
        std::cout << "[LLM Task] : Queue " << m_queue_llm_tasks.size() << std::endl;
    }
    // return handle;
}

void AsyncInternVL3Runner::worker_loop(
) {
    while (!m_stop) {
        std::vector<GenerateResult*> gen_results;
        {   // Vision : Queue -> Process
            std::lock_guard<std::mutex> lock(m_vis_queue_mutex);
            if (!m_queue_vis_tasks.empty()) {
                SharedVisGenHandle handle = m_queue_vis_tasks.front();

                std::lock_guard<std::mutex> map_lock(m_map_mutex);
                // if ((m_inflight_vis_tasks.size() < max_inflight_vis) && handle && (m_inflight_llm_tasks.size() < max_inflight_llm)) {
                if ((m_inflight_vis_tasks.size() < max_inflight_vis) && handle) {
                    m_queue_vis_tasks.pop_front();
                    
                    m_inflight_vis_tasks[handle->vis_task_id] = handle;
                    m_sync_runner.vis_engine.enqueue_extract_visual_features(
                        handle->visual_features.images, 
                        handle->visual_features.gen_config, 
                        handle);

                    std::cout << "[VIS Task] : Process " << m_inflight_vis_tasks.size() << " / " << max_inflight_vis << std::endl;
                } else {
                    // std::cout << std::endl;
                }
            }
        }
        {
            std::lock_guard<std::mutex> lock(m_map_mutex);
            for (auto it = m_inflight_vis_tasks.begin(); it != m_inflight_vis_tasks.end(); ++it) {
                auto& handle = it->second;
                if (handle->vis_finished.load()) {
                    if (handle->do_llm.load()) {
                        std::lock_guard<std::mutex> lock(m_llm_queue_mutex);
                        m_queue_llm_tasks.push_back(handle);
                        std::cout << "[LLM Task] : Queue " << m_queue_llm_tasks.size() << std::endl;
                    }
                    it = m_inflight_vis_tasks.erase(it);
                    break;
                }
            }
        }
        {   // LLM : Queue -> Process
            std::lock_guard<std::mutex> lock(m_llm_queue_mutex);

            if (!m_queue_llm_tasks.empty()) {
                SharedVisGenHandle handle = m_queue_llm_tasks.front();

                std::lock_guard<std::mutex> map_lock(m_map_mutex);
                if ((m_inflight_llm_tasks.size() < max_inflight_llm) && handle) {
                    m_queue_llm_tasks.pop_front();
                    
                    m_inflight_llm_tasks[handle->llm_task_id] = handle;
                    m_sync_runner.llm_engine.enqueue_generate_from_features(
                        handle->visual_features, 
                        handle->generate_result.user_prompt, 
                        handle->generate_result.gen_config, 
                        handle);

                    std::cout << "[LLM Task] : Process " << m_inflight_llm_tasks.size() << " / " << max_inflight_llm << std::endl;
                } else {
                    // std::cout << std::endl;
                }

            }
        }
        {
            std::lock_guard<std::mutex> map_lock(m_map_mutex);
            for (auto& [rid, handle] : m_inflight_llm_tasks) {
                gen_results.push_back(&(handle->generate_result));
            }
        }
        m_sync_runner.llm_engine.update_response(gen_results, 1000, false);

        {   // LLM : Done -> Removed
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