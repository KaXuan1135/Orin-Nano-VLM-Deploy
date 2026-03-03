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

    SharedGenHandle enqueue_generate(
        const std::vector<cv::Mat>& images, 
        const std::string& user_prompt,
        const GenerateConfig& gen_config
    ) override;

    SharedVisHandle enqueue_extract_visual_features(
        const std::vector<cv::Mat>& images,
        const GenerateConfig& gen_config
    ) override;

    SharedGenHandle enqueue_generate_from_features(
        const VisualFeatures& visual_features,
        const std::string& user_prompt,
        const GenerateConfig& gen_config
    ) override;

private:

    InternVL3Runner m_sync_runner;
    std::thread m_worker;
    std::atomic<bool> m_stop;
    std::condition_variable m_cv;
    std::unordered_map<uint64_t, SharedGenHandle> m_inflight_llm_tasks;

    void generate_listener_loop();

};

} // namespace trt_multimodal