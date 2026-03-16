#pragma once
#include <atomic>
#include <string>
#include <vector>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>

namespace trt_multimodal {

namespace ModelType {
    enum class Type {
        UNKNOWN = 0,
        INTERNVL3,
    };

    inline std::string to_string(Type type) {
        switch (type) {
            case Type::INTERNVL3: return "INTERNVL3";
            default:              return "UNKNOWN";
        }
    }

    inline std::string get_supported_model_types() {
        return "INTERNVL3";
    }
}

enum DataType {
    UNKNOWN = 0,
    FP32,
    FP16,
    BF16,
    INT8,
    UINT8,
    INT4,
    UINT4
};

struct TokenizerMetadata {
    int eos_id = -1;
    int pad_id = -1;
    int bos_id = -1;
};

struct AspectRatio {
    int width;
    int height;
};

struct ModelConfig {
    ModelType::Type model_type;
    std::string llm_engine_path = "";
    std::string vis_engine_path = "";
    std::string tokenizer_path = "";

    int32_t max_beam_width = 1;
    int32_t max_llm_batch = 1;
    int32_t max_vis_batch = 1;
    int32_t patch_token_size = 256;
    int32_t embedding_dim = 896;

    // TODO: more to add
    //最大 Batch Size。
    //最大 Context 长度。

};

struct GenerateConfig {
    std::string system_prompt = "You are an AI Model.";
    std::string image_prefix = "Image-$N$: ";
    std::string image_postfix = "\n";
    int32_t max_new_tokens = 512;
    int32_t top_k = 1;
    float top_p = 0.0f;
    float temperature = 0.2f;
    float repetition_penalty = 1.0f;
    bool streaming = false;

    // TODO: more to add
    int32_t min_patch = 1;
    int32_t max_patch = 1;
    int32_t patch_size = 448;
    bool use_thumbnail = false;

    //Profiling
    bool profiling = false;
};

struct VisualFeatures {

    GenerateConfig gen_config;
    std::vector<cv::Mat> images;

    std::shared_ptr<void> embeddings_ptr;
    std::vector<int32_t> image_patch_counts;
    DataType dtype = DataType::UNKNOWN;

    // Only Return if Generate Config :: profiling == true
    double vision_latency_ms = -1;    // 视觉特征提取耗时

    size_t total_patches() const {
        size_t total = 0;
        for (const int& count : image_patch_counts) total += count; 
        return total;
    }

};

struct GenerateResult {

    GenerateConfig gen_config;

    std::uint64_t request_id;
    std::uint64_t ttft_request_id;
    
    std::string system_prompt;
    std::string user_prompt;

    std::vector<int32_t> input_tokens;
    std::vector<std::vector<int32_t>> outputs_tokens; // beams of output tokens
    std::vector<std::string> outputs_text; // beams of output text

    mutable std::mutex data_mutex;
    std::vector<std::vector<int32_t>> last_outputs_token; // last output tokens (the user should clean it after reading), for streaming purpose
    std::vector<std::string> last_outputs_text;

    int32_t input_tokens_len() const {
        return input_tokens.size();
    }

    std::vector<int32_t> outputs_tokens_len() const {
        std::vector<int32_t> outputs_tokens_len;
        for (auto const& output_tokens : outputs_tokens) {
            outputs_tokens_len.push_back(output_tokens.size());            
        }
        return outputs_tokens_len;
    }

    std::atomic<bool> done_output{false}; // For Asynchonize output, determine if this result is complete.
    std::vector<bool> full_stops; // True if model did complete output for each beam, False if stop because of max_new tokens reach limit
    
    std::string error_msg;
    std::atomic<bool> has_error{false};

    // Only Return if Generate Config :: profiling == true
    std::chrono::high_resolution_clock::time_point start_gen;
    std::chrono::high_resolution_clock::time_point end_gen;
    std::chrono::high_resolution_clock::time_point start_ttft;
    std::chrono::high_resolution_clock::time_point end_ttft;
    std::atomic<bool> first_token_captured{false};

    // --- NEW: Add these to allow use in std::vector ---

    // 1. Default Constructor (Required for resize/emplace)
    GenerateResult() = default;

    // 2. Move Constructor
    // We manually move strings/vectors and load() the atomic values
    GenerateResult(GenerateResult&& other) noexcept {
        gen_config = std::move(other.gen_config);
        request_id = other.request_id;
        ttft_request_id = other.ttft_request_id;
        system_prompt = std::move(other.system_prompt);
        user_prompt = std::move(other.user_prompt);
        input_tokens = std::move(other.input_tokens);
        outputs_tokens = std::move(other.outputs_tokens);
        outputs_text = std::move(other.outputs_text);
        last_outputs_token = std::move(other.last_outputs_token);
        last_outputs_text = std::move(other.last_outputs_text);
        full_stops = std::move(other.full_stops);
        error_msg = std::move(other.error_msg);
        
        // Atomics must be loaded and stored
        done_output.store(other.done_output.load());
        has_error.store(other.has_error.load());
        first_token_captured.store(other.first_token_captured.load());

        start_gen = other.start_gen;
        end_gen = other.end_gen;
        start_ttft = other.start_ttft;
        end_ttft = other.end_ttft;
    }

    // 3. Move Assignment
    GenerateResult& operator=(GenerateResult&& other) noexcept {
        if (this != &other) {
            gen_config = std::move(other.gen_config);
            request_id = other.request_id;
            ttft_request_id = other.ttft_request_id;
            system_prompt = std::move(other.system_prompt);
            user_prompt = std::move(other.user_prompt);
            input_tokens = std::move(other.input_tokens);
            outputs_tokens = std::move(other.outputs_tokens);
            outputs_text = std::move(other.outputs_text);
            last_outputs_token = std::move(other.last_outputs_token);
            last_outputs_text = std::move(other.last_outputs_text);
            full_stops = std::move(other.full_stops);
            error_msg = std::move(other.error_msg);
            
            done_output.store(other.done_output.load());
            has_error.store(other.has_error.load());
            first_token_captured.store(other.first_token_captured.load());

            start_gen = other.start_gen;
            end_gen = other.end_gen;
            start_ttft = other.start_ttft;
            end_ttft = other.end_ttft;
        }
        return *this;
    }

    // 4. Explicitly Delete Copy (Since atomics can't be copied)
    GenerateResult(const GenerateResult&) = delete;
    GenerateResult& operator=(const GenerateResult&) = delete;



    double generation_latency_ms() const {
        if (!gen_config.profiling) return 0.0;
        return std::chrono::duration<double, std::milli>(end_gen - start_gen).count();
    }

    double time_to_first_token_ms() const {
        if (!gen_config.profiling) return 0.0;
        return std::chrono::duration<double, std::milli>(end_ttft - start_ttft).count();
    }


    std::vector<int32_t> total_tokens() const {
        std::vector<int32_t> total_tokens;
        for (const auto& output_tokens_len : outputs_tokens_len()) {
            total_tokens.push_back(output_tokens_len + input_tokens_len());
        }
        return total_tokens;
    }
    
    // First Beams Inference Speed
    double tokens_per_second() const {
        if (generation_latency_ms() <= 0) return 0.0;
        return (outputs_tokens_len()[0] / generation_latency_ms()) * 1000.0;
    }

    // Whole System Throughput
    double system_throughput() const {
        double lat = generation_latency_ms();
        if (lat <= 0) return 0.0;
        auto lens = outputs_tokens_len(); 
        return (std::accumulate(lens.begin(), lens.end(), 0.0) / lat) * 1000.0;
    }

};

//Asynchronous
struct VisGenHandle {

    std::uint64_t vis_task_id;
    std::uint64_t llm_task_id;

    std::atomic<bool> vis_finished{false};
    std::atomic<bool> gen_finished{false};

    std::atomic<bool> do_vis{false};
    std::atomic<bool> do_llm{false};

    VisualFeatures visual_features;
    GenerateResult generate_result;

    std::vector<std::shared_ptr<VisGenHandle>> prev_handles; //temp solution

    std::vector<std::string> pop_last_outputs_text() {
        std::lock_guard<std::mutex> lock(generate_result.data_mutex);
        std::vector<std::string> result = generate_result.last_outputs_text; // 拷贝一份

        for (auto& beam : generate_result.last_outputs_token) {
            beam.clear();
        }

        generate_result.last_outputs_text.clear();
        generate_result.last_outputs_text.resize(result.size()); 
        return result;
    }

};
using SharedVisGenHandle = std::shared_ptr<VisGenHandle>;

} // namespace trt_multimodal