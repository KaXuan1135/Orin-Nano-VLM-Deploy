#pragma once
#include <mutex>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <condition_variable>

#include "TrtMultimodalRunner/Types.hpp" 
#include "TrtMultimodalRunner/IAsyncMultimodalRunner.hpp"
#include "TrtMultimodalRunner/InternVL3/InternVL3Runner.hpp"
#include "TrtMultimodalRunner/InternVL3/InternVL3LLMEngine.hpp"
#include "TrtMultimodalRunner/InternVL3/InternVL3VisionEngine.hpp"

namespace trt_multimodal {

class AsyncInternVL3Runner : public IAsyncMultimodalRunner {
public:

    // User should only create this class by calling IAsyncMultimodalRunner create function
    explicit AsyncInternVL3Runner(const ModelConfig& config); 

    ~AsyncInternVL3Runner() override;

    SharedVisGenHandle enqueue_generate(
        const std::vector<cv::Mat>& images, 
        const std::string& user_prompt,
        const GenerateConfig& gen_config
    ) override;

    SharedVisGenHandle enqueue_extract_visual_features(
        const std::vector<cv::Mat>& images,
        const GenerateConfig& gen_config
    ) override;

    SharedVisGenHandle enqueue_generate_from_features(
        const VisualFeatures& visual_features,
        const std::string& user_prompt,
        const GenerateConfig& gen_config
    ) override;

private:

    InternVL3Runner m_sync_runner;
    std::thread m_worker;

    std::atomic<bool> m_stop;
    std::condition_variable m_cv;

    mutable std::mutex m_map_mutex;
    mutable std::mutex m_vis_queue_mutex;
    mutable std::mutex m_llm_queue_mutex;
    std::atomic<uint64_t> vis_rid{0};
    std::atomic<uint64_t> llm_rid{0};
    size_t max_inflight_vis = 5;  // TODO, move to arguments
    size_t max_inflight_llm = 20; // TODO, move to arguments
    std::deque<SharedVisGenHandle> m_queue_vis_tasks;
    std::deque<SharedVisGenHandle> m_queue_llm_tasks;
    std::unordered_map<uint64_t, SharedVisGenHandle> m_inflight_vis_tasks;
    std::unordered_map<uint64_t, SharedVisGenHandle> m_inflight_llm_tasks;

    void worker_loop();

};

} // namespace trt_multimodal