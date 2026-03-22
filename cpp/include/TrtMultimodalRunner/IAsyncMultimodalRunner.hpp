#pragma once
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <iomanip>

#include "TrtMultimodalRunner/Types.hpp"

namespace cv { class Mat; }

namespace trt_multimodal {

class IAsyncMultimodalRunner {
public:
    virtual ~IAsyncMultimodalRunner() = default;

    static std::unique_ptr<IAsyncMultimodalRunner> create(
        const ModelConfig& model_config
    );

    static SharedVisGenHandle create_handle(
        const GenerateConfig& gen_config,
        const std::string& user_prompt,
        const std::vector<cv::Mat>& images = {}, 
        const std::vector<SharedVisGenHandle>& history_handles = {}
    ) {
        SharedVisGenHandle handle = std::make_shared<VisGenHandle>();
        handle->gen_config = gen_config;
        handle->history_handles = history_handles;
        handle->generate_result.user_prompt = user_prompt;
        handle->visual_features.images = images;

        return handle;
    }

    virtual void enqueue_generate(
        SharedVisGenHandle& handle
    ) = 0; 

    virtual void enqueue_extract_visual_features(
        SharedVisGenHandle& handle
    ) = 0;

    virtual void enqueue_generate_from_features(
        SharedVisGenHandle& handle
    ) = 0;

    static void print_benchmark(
        std::chrono::time_point<std::chrono::high_resolution_clock> overall_start,
        std::chrono::time_point<std::chrono::high_resolution_clock> overall_end,
        std::vector<SharedVisGenHandle>& handles
    ) {
        if (handles.empty()) return;

        size_t request_num = handles.size();
        std::vector<double> vis_qt_list(request_num), vis_proc_list(request_num);
        std::vector<double> llm_qt_list(request_num), llm_ttft_list(request_num), llm_tps_list(request_num);
        double total_tokens = 0;

        for (size_t i = 0; i < request_num; ++i) {
            auto& res = handles[i];

            vis_qt_list[i] = std::chrono::duration<double>(res->visual_features.end_queue - res->visual_features.start_queue).count();
            vis_proc_list[i] = std::chrono::duration<double>(res->visual_features.end_proc - res->visual_features.start_proc).count();

            llm_qt_list[i] = res->generate_result.queue_latency();
            llm_ttft_list[i] = res->generate_result.time_to_first_token();
            llm_tps_list[i] = res->generate_result.tokens_per_second();

            auto lens = res->generate_result.outputs_tokens_len();
            for (size_t j = 0; j < lens.size(); ++j) total_tokens += lens[j];
        }

        double total_elapsed_s = std::chrono::duration<double>(overall_end - overall_start).count();

        auto get_p = [](std::vector<double> v, double p) {
            if (v.empty()) return 0.0;
            size_t idx = static_cast<size_t>(std::ceil(p * v.size())) - 1;
            idx = std::min(idx, v.size() - 1);
            std::nth_element(v.begin(), v.begin() + idx, v.end());
            return v[idx];
        };

        auto avg = [](const std::vector<double>& v) {
            return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        };

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "         Overall Performance Summary (n=" << request_num << ")" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Total Elapsed Time    : " << std::setw(8) << total_elapsed_s << " s" << std::endl;
        std::cout << "Overall Throughput    : " << std::setw(8) << (total_tokens / total_elapsed_s) << " tok/s" << std::endl;
        std::cout << "Avg Output Tokens/Req : " << std::setw(8) << (total_tokens / request_num) << " tok" << std::endl;

        std::cout << "\nPer-Request Summary (Latencies in seconds)" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << "  Metric           |    Average    |      P99      |" << std::endl;
        std::cout << "-------------------|---------------|---------------|" << std::endl;
        
        // Vision
        std::cout << "  Vis Queue (s)    |   " << std::setw(9) << avg(vis_qt_list) 
                << "   |   " << std::setw(9) << get_p(vis_qt_list, 0.99) << "   |" << std::endl;
        std::cout << "  Vis Proc  (s)    |   " << std::setw(9) << avg(vis_proc_list) 
                << "   |   " << std::setw(9) << get_p(vis_proc_list, 0.99) << "   |" << std::endl;
        std::cout << "-------------------|---------------|---------------|" << std::endl;

        // LLM
        std::cout << "  LLM Queue (s)    |   " << std::setw(9) << avg(llm_qt_list) 
                << "   |   " << std::setw(9) << get_p(llm_qt_list, 0.99) << "   |" << std::endl;
        std::cout << "  LLM TTFT  (s)    |   " << std::setw(9) << avg(llm_ttft_list) 
                << "   |   " << std::setw(9) << get_p(llm_ttft_list, 0.99) << "   |" << std::endl;
        std::cout << "  LLM TPS (tok/s)  |   " << std::setw(9) << avg(llm_tps_list) 
                << "   |   " << std::setw(9) << get_p(llm_tps_list, 0.01) << "   |" << std::endl;
        std::cout << std::string(60, '=') << std::endl << std::endl;
    }

protected:

    static void monitor_update(size_t vis_q, size_t vis_p, size_t llm_q, size_t llm_p, int max_vis, int max_llm) {
        std::cerr << "\033[s"; 

        std::cerr << "\033[2;1H";

        std::cerr << "\033[1;36m[SYSTEM MONITOR]\033[0m "
                  << "VIS Queue: " << std::setw(2) << vis_q << " | "
                  << "VIS Proc: [" << std::setw(2) << vis_p << "/" << max_vis << "] | "
                  << "LLM Queue: " << std::setw(2) << llm_q << " | "
                  << "LLM Proc: [" << std::setw(2) << llm_p << "/" << max_llm << "]"
                  << "\033[K";

        std::cerr << "\n\033[1;30m" << std::string(80, '-') << "\033[0m\033[K";
        std::cerr << "\033[u" << std::flush;
    }

};

} // namespace trt_multimodal