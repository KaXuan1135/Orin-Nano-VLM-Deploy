// #include "Utils/Monitor.hpp"
#include "TrtMultimodalRunner/InternVL3/AsyncInternVL3Runner.hpp"

namespace trt_multimodal {

AsyncInternVL3Runner::AsyncInternVL3Runner(const ModelConfig& config) 
    :m_sync_runner(config), m_stop(false)
{
    m_worker = std::thread(&AsyncInternVL3Runner::worker_loop, this);
    m_sync_runner.vis_engine->init_static_pool(max_inflight_vis);
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

void AsyncInternVL3Runner::enqueue_generate(
    SharedVisGenHandle& handle
) {
    enqueue_extract_visual_features(handle);
    handle->llm_task_id = llm_rid++;
    handle->do_llm.store(true);
}

void AsyncInternVL3Runner::enqueue_extract_visual_features(
    SharedVisGenHandle& handle
) {
    handle->vis_task_id = vis_rid++;
    handle->do_vis.store(true);
    {
        std::lock_guard<std::mutex> lock(m_vis_queue_mutex);
        m_queue_vis_tasks.push_back(handle);
        handle->visual_features.start_queue = std::chrono::high_resolution_clock::now();
    }
}

void AsyncInternVL3Runner::enqueue_generate_from_features(
    SharedVisGenHandle& handle
) {
    handle->llm_task_id = llm_rid++;
    handle->do_llm.store(true);
    {
        std::lock_guard<std::mutex> lock(m_llm_queue_mutex);
        m_queue_llm_tasks.push_back(handle);
    }
}

void AsyncInternVL3Runner::worker_loop(
) {
    while (!m_stop) {
        std::vector<SharedVisGenHandle> to_update_handles;
        
        // SharedVisGenHandle vis_task_to_launch = nullptr;
        {   // Vision : Queue -> Process
            std::lock_guard<std::mutex> lock(m_vis_queue_mutex);
            if (!m_queue_vis_tasks.empty()) {
                SharedVisGenHandle handle = m_queue_vis_tasks.front();

                std::lock_guard<std::mutex> map_lock(m_map_mutex);
                if ((m_inflight_vis_tasks.size() < max_inflight_vis) && handle && (m_inflight_llm_tasks.size() < max_inflight_llm)) {
                // if ((m_inflight_vis_tasks.size() < max_inflight_vis) && handle) {
                    m_queue_vis_tasks.pop_front();
                    // vis_task_to_launch = handle;

                    handle->visual_features.end_queue = std::chrono::high_resolution_clock::now();
                    m_inflight_vis_tasks[handle->vis_task_id] = handle;
                    auto s = std::chrono::high_resolution_clock::now();
                    m_sync_runner.vis_engine->enqueue_extract_visual_features(handle);
                    auto e = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();

                    std::cout << "[DEBUG] Enqueue function call cost: " << duration << " ms" << std::endl;
                    
                }
            }
        }
        // if (vis_task_to_launch) {
        //     vis_task_to_launch->visual_features.end_queue = std::chrono::high_resolution_clock::now();
        //     m_inflight_vis_tasks[vis_task_to_launch->vis_task_id] = vis_task_to_launch;
        //     auto s = std::chrono::high_resolution_clock::now();
        //     m_sync_runner.vis_engine->enqueue_extract_visual_features(vis_task_to_launch);
        //     auto e = std::chrono::high_resolution_clock::now();
        //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();

        //     std::cout << "[DEBUG] Enqueue function call cost: " << duration << " ms" << std::endl;
        // }
        {   // Vision : Done -> LLM : Queue
            std::lock_guard<std::mutex> lock(m_map_mutex);
            for (auto it = m_inflight_vis_tasks.begin(); it != m_inflight_vis_tasks.end(); ++it) {
                auto& handle = it->second;
                if (handle->vis_finished.load()) {
                    if (handle->do_llm.load()) {
                        std::lock_guard<std::mutex> lock(m_llm_queue_mutex);
                        handle->generate_result.start_queue = std::chrono::high_resolution_clock::now();
                        m_queue_llm_tasks.push_back(handle);
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
                    
                    handle->generate_result.end_queue = std::chrono::high_resolution_clock::now();
                    m_inflight_llm_tasks[handle->llm_task_id] = handle;
                    m_sync_runner.llm_engine->enqueue_generate_from_features(handle);
                }
            }
        }
        {
            std::lock_guard<std::mutex> map_lock(m_map_mutex);
            for (auto& [rid, handle] : m_inflight_llm_tasks) {
                to_update_handles.push_back(handle);
            }
        }
        m_sync_runner.llm_engine->update_response(to_update_handles, 1000, false);
        {   // LLM : Done -> Removed
            std::lock_guard<std::mutex> lock(m_map_mutex);
            for (auto it = m_inflight_llm_tasks.begin(); it != m_inflight_llm_tasks.end(); ) {
                auto& handle = it->second;
                
                // if (handle->generate_result.done_output.load()) {
                if (handle->generate_result.done_output) {
                    handle->gen_finished.store(true);
                    it = m_inflight_llm_tasks.erase(it);
                } else {
                    ++it;
                }
            }
        }
        {   // Monitor
            std::lock_guard<std::mutex> lock_v(m_vis_queue_mutex);
            std::lock_guard<std::mutex> lock_l(m_llm_queue_mutex);
            std::lock_guard<std::mutex> lock_m(m_map_mutex);

            monitor_update(
                m_queue_vis_tasks.size(),
                m_inflight_vis_tasks.size(),
                m_queue_llm_tasks.size(),
                m_inflight_llm_tasks.size(),
                max_inflight_vis,
                max_inflight_llm
            );
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}


} // namespace trt_multimodal