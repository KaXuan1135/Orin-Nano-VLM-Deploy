#pragma once
#include <string>
#include <vector>
#include <memory>
#include <numeric>

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

    std::uint64_t request_id;
    
    std::string system_prompt;
    std::string user_prompt;
    std::vector<std::string> outputs_text; // beams of output text

    int32_t input_tokens_len;
    std::vector<int32_t> outputs_tokens_len;

    bool done_output = false; // For Asynchonize output, determine if this result is complete.
    std::vector<bool> full_stops; // True if model did complete output for each beam, False if stop because of max_new tokens reach limit
    
    // Only Return if Generate Config :: profiling == true
    double generation_latency_ms = -1; // 文本生成耗时
    double time_to_first_token_ms = -1;

    std::vector<int32_t> total_tokens() const {
        std::vector<int32_t> total_tokens;
        for (const auto& output_tokens_len : outputs_tokens_len) {
            total_tokens.push_back(output_tokens_len + input_tokens_len);
        }
        return total_tokens;
    }
    
    // First Beams Inferenc Speed
    double tokens_per_second() const {
        if (generation_latency_ms <= 0) return 0.0;
        return (outputs_tokens_len[0] / generation_latency_ms) * 1000.0;
    }

    // Whole System Throughput
    double system_throughput() const {
        if (generation_latency_ms <= 0) return 0.0;
        return (std::accumulate(outputs_tokens_len.begin(), outputs_tokens_len.end(), 0.0) / generation_latency_ms) * 1000.0;
    }

};

} // namespace trt_multimodal