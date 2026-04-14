#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <fstream>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <nlohmann/json.hpp>

#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "TrtMultimodalRunner/InternVL3/InternVL3LLMEngine.hpp"

namespace tle = tensorrt_llm::executor;
using json = nlohmann::json;

namespace trt_multimodal {

tle::Request create_request_from_dict(
    const std::vector<int32_t> input_ids,
    const std::vector<VisualFeatures>& vis_features,
    const ModelConfig& m_config,
    const GenerateConfig& gen_config,
    const TokenizerMetadata& metadata,
    void* d_combined_ptr
) {
    
    tle::SamplingConfig sampling_config(m_config.max_beam_width);
    sampling_config.setTopK(gen_config.top_k);
    sampling_config.setTopP(gen_config.top_p ? gen_config.top_p > 0.0f : 1.0f); 
    sampling_config.setTemperature(gen_config.temperature);
    sampling_config.setRepetitionPenalty(gen_config.repetition_penalty);

    size_t total_all_patches = 0;
    for (const auto& vf : vis_features) {
        total_all_patches += vf.total_patches();
    }
    
    size_t total_tokens = total_all_patches * m_config.patch_tokens;
    size_t embedding_dim = m_config.embedding_dim;
    
    size_t element_size = 2; // assume bf16 or fp16
    size_t total_bytes = total_tokens * embedding_dim * element_size;
    
    cudaMalloc(&d_combined_ptr, total_bytes);

    size_t offset_bytes = 0;
    for (const auto& vf : vis_features) {
        size_t current_bytes = vf.total_patches() * m_config.patch_tokens * embedding_dim * element_size;
        cudaMemcpy((char*)d_combined_ptr + offset_bytes, vf.embeddings_ptr.get(), current_bytes, cudaMemcpyDeviceToDevice);
        offset_bytes += current_bytes;
    }

    tle::Tensor embedding = tle::Tensor::of(
        tle::DataType::kBF16,
        d_combined_ptr,
        tle::Shape{
            static_cast<long int>(total_tokens), 
            static_cast<long int>(m_config.embedding_dim)
        }
    );

    std::optional<tle::PromptTuningConfig> pTuningConfig = tle::PromptTuningConfig(embedding);

    tle::Request request(
        input_ids,                             // 1. VecTokens
        gen_config.max_new_tokens,             // 2. SizeType32
        gen_config.streaming,                  // 3. bool streaming
        sampling_config,                       // 4. const SamplingConfig&
        tle::OutputConfig(),                   // 5. const OutputConfig&
        metadata.eos_id,                       // 6. std::optional<int> endId
        metadata.pad_id,                       // 7. std::optional<int> padId
        std::nullopt,                          // 8. badWords
        std::nullopt,                          // 9. stopWords
        std::nullopt,                          // 10. std::optional<Tensor> embeddingBias
        std::nullopt,                          // 11. std::optional<ExternalDraftTokensConfig>
        pTuningConfig,                         // 12. PromptTuningConfig (图像特征)
        std::nullopt,                          // 13. std::optional<LoraConfig>
        std::nullopt,                          // 14. std::optional<std::string> lookaheadConfig
        std::nullopt,                          // 15. std::optional<std::vector<int>> taskVocabSize
        std::nullopt,                          // 16. std::optional<uint64_t> schedulerPolicyData
        false,                                 // 17. bool returnAllGeneratedTokens
        0.0f                                   // 18. PriorityType priority 
    );

    return request;
}

tle::Request create_request_from_dict_async(
    const std::vector<int32_t> input_ids,
    const std::vector<VisualFeatures>& vis_features,
    const ModelConfig& m_config,
    const GenerateConfig& gen_config,
    const TokenizerMetadata& metadata,
    const cudaStream_t& m_stream,
    void* d_combined_ptr
) {
    
    tle::SamplingConfig sampling_config(m_config.max_beam_width);
    sampling_config.setTopK(gen_config.top_k);
    sampling_config.setTopP(gen_config.top_p ? gen_config.top_p > 0.0f : 1.0f); 
    sampling_config.setTemperature(gen_config.temperature);
    sampling_config.setRepetitionPenalty(gen_config.repetition_penalty);

    size_t total_tokens;
    if (vis_features.size() == 1) {
        d_combined_ptr = vis_features[0].embeddings_ptr.get();
        total_tokens = vis_features[0].total_patches() * m_config.patch_tokens;
    } else {
        size_t total_all_patches = 0;
        for (const auto& vf : vis_features) {
            total_all_patches += vf.total_patches();
        }
        
        total_tokens = total_all_patches * m_config.patch_tokens;
        size_t embedding_dim = m_config.embedding_dim;
        
        size_t element_size = 2; // assume bf16 or fp16
        size_t total_bytes = total_tokens * embedding_dim * element_size;
        
        cudaMallocAsync(&d_combined_ptr, total_bytes, m_stream);
        
        size_t offset_bytes = 0;
        for (const auto& vf : vis_features) {
            size_t current_bytes = vf.total_patches() * m_config.patch_tokens * embedding_dim * element_size;
            cudaMemcpyAsync((char*)d_combined_ptr + offset_bytes, vf.embeddings_ptr.get(), current_bytes, cudaMemcpyDeviceToDevice, m_stream);
            offset_bytes += current_bytes;
        }
    }

    tle::Tensor embedding = tle::Tensor::of(
        tle::DataType::kBF16,
        d_combined_ptr,
        tle::Shape{
            static_cast<long int>(total_tokens), 
            static_cast<long int>(m_config.embedding_dim)
        }
    );

    std::optional<tle::PromptTuningConfig> pTuningConfig = tle::PromptTuningConfig(embedding);

    tle::Request request(
        input_ids,                             // 1. VecTokens
        gen_config.max_new_tokens,             // 2. SizeType32
        gen_config.streaming,                  // 3. bool streaming
        sampling_config,                       // 4. const SamplingConfig&
        tle::OutputConfig(),                   // 5. const OutputConfig&
        metadata.eos_id,                       // 6. std::optional<int> endId
        metadata.pad_id,                       // 7. std::optional<int> padId
        std::nullopt,                          // 8. badWords
        std::nullopt,                          // 9. stopWords
        std::nullopt,                          // 10. std::optional<Tensor> embeddingBias
        std::nullopt,                          // 11. std::optional<ExternalDraftTokensConfig>
        pTuningConfig,                         // 12. PromptTuningConfig (图像特征)
        std::nullopt,                          // 13. std::optional<LoraConfig>
        std::nullopt,                          // 14. std::optional<std::string> lookaheadConfig
        std::nullopt,                          // 15. std::optional<std::vector<int>> taskVocabSize
        std::nullopt,                          // 16. std::optional<uint64_t> schedulerPolicyData
        false,                                 // 17. bool returnAllGeneratedTokens
        0.0f                                   // 18. PriorityType priority 
    );

    return request;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Failed to open tokenizer file: " << path << std::endl;
        return "";
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string strip(const std::string& s) {
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start)) start++;
    auto end = s.end();
    do { end--; } while (std::distance(start, end) > 0 && std::isspace(*end));
    return std::string(start, end + 1);
}

TokenizerMetadata get_tokenizer_ids(const std::string& path) {
    std::ifstream f(path);
    json data = json::parse(f);
    TokenizerMetadata meta;

    // 优先从 added_tokens 找
    if (data.contains("added_tokens")) {
        for (auto& t : data["added_tokens"]) {
            std::string content = t["content"];
            if (content == "<|im_end|>") meta.eos_id = t["id"];
            if (content == "<|endoftext|>") meta.pad_id = t["id"];
        }
    }

    // 如果 pad_id 还是默认或没找到，去基础 vocab 找
    if (data.contains("model") && data["model"].contains("vocab")) {
        auto const& vocab = data["model"]["vocab"];
        
        // 检查 <|endoftext|> 是否在基础词表里
        if (vocab.contains("<|endoftext|>")) {
            meta.pad_id = vocab["<|endoftext|>"];
        }
        
        // 如果需要找普通的 <|im_end|> 也可以在这里兜底
        if (vocab.contains("<|im_end|>")) {
            meta.eos_id = vocab["<|im_end|>"];
        }
    }

    return meta;
}

InternVL3LLMEngine::InternVL3LLMEngine(
    const ModelConfig& config,
    const cudaStream_t& stream
): m_config(config), m_stream(stream) {

    initTrtLlmPlugins();

    tensorrt_llm::executor::KvCacheConfig kvCacheConfig;
    kvCacheConfig.setFreeGpuMemoryFraction(m_config.kv_cache_reserved_space); 
    kvCacheConfig.setEnableBlockReuse(true);

    tensorrt_llm::executor::ExecutorConfig executorConfig(m_config.max_beam_width);
    executorConfig.setKvCacheConfig(kvCacheConfig);

    tensorrt_llm::executor::SchedulerConfig schedulerConfig(
        tensorrt_llm::executor::CapacitySchedulerPolicy::kMAX_UTILIZATION,
        tensorrt_llm::executor::ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED
    );

    executorConfig.setSchedulerConfig(schedulerConfig);

    llm_executor = std::make_unique<tensorrt_llm::executor::Executor>(
        m_config.llm_engine_path, 
        tensorrt_llm::executor::ModelType::kDECODER_ONLY,
        executorConfig
    );

    tokenizer = tokenizers::Tokenizer::FromBlobJSON(read_file(m_config.tokenizer_path));
    metadata = get_tokenizer_ids(m_config.tokenizer_path);
}

std::vector<std::string> InternVL3LLMEngine::decode_outputs(
    const std::vector<std::vector<int32_t>> beams_tokens,
    size_t input_len
) {
    std::vector<std::string> results(m_config.max_beam_width);

    for (int m = 0; m < m_config.max_beam_width; ++m) {
        const auto& current_beam = beams_tokens[m];

        results[m] = "";
        if (current_beam.size() > input_len) {
            std::vector<int32_t> output_ids(
                current_beam.begin() + input_len, 
                current_beam.end()
            );
            // results[m] = strip(tokenizer->Decode(output_ids));
            results[m] = tokenizer->Decode(output_ids);
        }
    }

    return results;
}

void InternVL3LLMEngine::generate_from_features(
    std::vector<SharedVisGenHandle> handles
) {
    size_t batch_size = handles.size();

    for (size_t b = 0; b < batch_size; ++b) {
        enqueue_generate_from_features(handles[b]);
    }

    // 3. Multi-request polling loop
    bool all_done = false;
    while (!all_done) {
        all_done = true;
        std::vector<SharedVisGenHandle> to_update_handles;
        
        for (size_t b = 0; b < batch_size; ++b) {
            if(!handles[b]->generate_result.done_output) {
                to_update_handles.push_back(handles[b]);
                all_done = false; 
            }
        }
        if (!to_update_handles.empty()) {
            update_response(to_update_handles, 1000, false);
            // Check for errors in the batch
            for (auto handle : to_update_handles) {
                auto res = &(handle->generate_result);
                if (res->has_error) {
                    throw std::runtime_error("TRT-LLM Error for Request " + 
                        std::to_string(res->request_id) + ": " + res->error_msg);
                }
            }
        }
    }

    // 4. Post-generation Profiling (TTFT estimation if needed)
    // Note: This logic repeats the inference for 1 token to measure TTFT overhead specifically.
    for (size_t b = 0; b < batch_size; ++b) {
        const auto& config = handles[b]->gen_config;
        if (config.profiling && !config.streaming) {
            GenerateConfig ttft_config = handles[b]->gen_config; // A new one
            ttft_config.max_new_tokens = 1;

            tle::Request ttft_req = create_request_from_dict(
                handles[b]->generate_result.input_tokens,
                {handles[b]->visual_features},
                m_config,
                ttft_config,
                metadata,
                handles[b]->generate_result.image_embeddings_gpu_ptr
            );

            handles[b]->generate_result.start_ttft = std::chrono::high_resolution_clock::now();
            handles[b]->generate_result.ttft_request_id = llm_executor->enqueueRequest(ttft_req);
            // Reset capture flag for the profiling run
            handles[b]->generate_result.first_token_captured = false; 

            while (!handles[b]->generate_result.first_token_captured) {
                std::vector<SharedVisGenHandle> to_update_handles = {handles[b]};
                update_response(to_update_handles, 1000, true);
            }
        }
    }
}

void InternVL3LLMEngine::enqueue_generate_from_features(
    SharedVisGenHandle& handle
) {

    const auto& config = handle->gen_config;

    std::vector<int32_t> input_ids;
    size_t cur_fake_id = tokenizer->GetVocabSize();
    std::vector<VisualFeatures> all_vis_feats;
    std::string post_prompt = "\n" + handle->generate_result.user_prompt + "<|im_end|>\n<|im_start|>assistant\n";
    if (handle->history_handles.size() == 0) {
        all_vis_feats.push_back(handle->visual_features);
        std::string pre_prompt = "<|im_start|>system\n" + config.system_prompt + "<|im_end|>\n<|im_start|>user\n";

        auto pre_tokens = tokenizer->Encode(pre_prompt);
        for (auto tid : pre_tokens) input_ids.push_back(static_cast<int32_t>(tid));

        for (int i = 0; i < handle->visual_features.image_patch_counts.size(); ++i) {
            int width = (m_config.max_num_frames > 0) ? static_cast<int>(std::log10(m_config.max_num_frames)) + 1 : 1;
            std::string prefix = config.image_prefix;
            std::string placeholder = "$N$";

            std::ostringstream oss;
            oss << std::setw(width) << std::setfill('0') << (i + 1);
            std::string formatted_index = oss.str();

            size_t pos = prefix.find(placeholder);
            if (pos != std::string::npos) {
                prefix.replace(pos, placeholder.length(), formatted_index);
            }

            auto pre_img_tokens = tokenizer->Encode(prefix);
            for (auto tid : pre_img_tokens) input_ids.push_back(static_cast<int32_t>(tid));

            for (int j = 0; j < m_config.patch_tokens * handle->visual_features.image_patch_counts[i]; ++j) {
                input_ids.push_back(cur_fake_id + j);
            }
            cur_fake_id += m_config.patch_tokens * handle->visual_features.image_patch_counts[i];

            auto post_img_tokens = tokenizer->Encode(config.image_postfix);
            for (auto tid : post_img_tokens) input_ids.push_back(static_cast<int32_t>(tid));
        }

        auto post_tokens = tokenizer->Encode(post_prompt);
        for (auto tid : post_tokens) input_ids.push_back(static_cast<int32_t>(tid));

    } else {
        size_t count = 0;
        std::string pre_prompt = "<|im_end|>\n<|im_start|>user\n";

        auto pre_tokens = tokenizer->Encode(pre_prompt);
        count += pre_tokens.size();

        for (int i = 0; i < handle->visual_features.image_patch_counts.size(); ++i) {
            int width = (m_config.max_num_frames > 0) ? static_cast<int>(std::log10(m_config.max_num_frames)) + 1 : 1;
            std::string prefix = config.image_prefix;
            std::string placeholder = "$N$";

            std::ostringstream oss;
            oss << std::setw(width) << std::setfill('0') << (i + 1);
            std::string formatted_index = oss.str();

            size_t pos = prefix.find(placeholder);
            if (pos != std::string::npos) {
                prefix.replace(pos, placeholder.length(), formatted_index);
            }

            auto pre_img_tokens = tokenizer->Encode(prefix);
            count += pre_img_tokens.size();

            count += m_config.patch_tokens * handle->visual_features.image_patch_counts[i];

            auto post_img_tokens = tokenizer->Encode(config.image_postfix);
            count += post_img_tokens.size();
        }

        auto post_tokens = tokenizer->Encode(post_prompt);
        count += post_tokens.size();
        
        std::cout << "Current Input Lens " << count << " / " << m_config.max_input_len << std::endl;

        auto sys_tokens = tokenizer->Encode("<|im_start|>system\n" + config.system_prompt);
        size_t system_count = sys_tokens.size();

        size_t accm_count = count + system_count;
        size_t img_count = handle->visual_features.total_patches();
        int history_chats_start_from = handle->history_handles.size();
        size_t retained_chats = 0;

        for (int i = handle->history_handles.size() - 1; i >= 0; --i) {

            auto const& cur_handle = handle->history_handles[i];
            

            std::string cur_req_prompt = "<|im_start|>user\n" + cur_handle->generate_result.user_prompt + "<|im_end|>\n";
            cur_req_prompt += "<|im_start|>assistant\n" + cur_handle->generate_result.outputs_text[0] + "<|im_end|>\n";
            std::vector<int32_t> cur_req_prompt_token = tokenizer->Encode(cur_req_prompt);

            size_t cur_count = cur_req_prompt_token.size();
            size_t cur_img_count = cur_handle->visual_features.total_patches();

            for (int j = 0; j < cur_handle->visual_features.image_patch_counts.size(); ++j) {
                int width = (m_config.max_num_frames > 0) ? static_cast<int>(std::log10(m_config.max_num_frames)) + 1 : 1;
                std::string prefix = config.image_prefix;
                std::string placeholder = "$N$";

                std::ostringstream oss;
                oss << std::setw(width) << std::setfill('0') << (i + 1);
                std::string formatted_index = oss.str();

                size_t pos = prefix.find(placeholder);
                if (pos != std::string::npos) {
                    prefix.replace(pos, placeholder.length(), formatted_index);
                }

                auto pre_img_tokens = tokenizer->Encode(prefix);
                cur_count += pre_img_tokens.size();

                cur_count += m_config.patch_tokens * cur_handle->visual_features.image_patch_counts[j];

                auto post_img_tokens = tokenizer->Encode(config.image_postfix);
                cur_count += post_img_tokens.size();
            }

            if (accm_count + cur_count > m_config.max_input_len || 
                img_count + cur_img_count > m_config.max_num_frames
            ) {
                std::cout << "[Dynamic Window] Limit reached at chat index " << i 
                    << ". Dropping oldest " << (i + 1) << " chats." << std::endl;

                break;
            } else {
                history_chats_start_from = i;
                retained_chats++;
                accm_count += cur_count;
                img_count += cur_img_count;
            }
        }

        // Start Constructing Whole input
        cur_fake_id = tokenizer->GetVocabSize();
        for (auto tid : sys_tokens) input_ids.push_back(static_cast<int32_t>(tid));

        int passed_in_images_count = 0;
        for (int i = history_chats_start_from; i < handle->history_handles.size(); ++i) {
            auto const& cur_handle = handle->history_handles[i];
            all_vis_feats.push_back(cur_handle->visual_features);

            std::string cur_req_pre_prompt = "<|im_start|>user\n";
            for (auto tid : tokenizer->Encode(cur_req_pre_prompt)) input_ids.push_back(static_cast<int32_t>(tid));

            for (int j = 0; j < cur_handle->visual_features.image_patch_counts.size(); ++j) {
                int width = (m_config.max_num_frames > 0) ? static_cast<int>(std::log10(m_config.max_num_frames)) + 1 : 1;
                std::string prefix = config.image_prefix;
                std::string placeholder = "$N$";

                std::ostringstream oss;
                oss << std::setw(width) << std::setfill('0') << (j + 1);
                std::string formatted_index = oss.str();

                size_t pos = prefix.find(placeholder);
                if (pos != std::string::npos) {
                    prefix.replace(pos, placeholder.length(), formatted_index);
                }

                auto pre_img_tokens = tokenizer->Encode(prefix);
                for (auto tid : pre_img_tokens) input_ids.push_back(static_cast<int32_t>(tid));

                for (int k = 0; k < m_config.patch_tokens * cur_handle->visual_features.image_patch_counts[j]; ++k) {
                    input_ids.push_back(cur_fake_id + k);
                }
                cur_fake_id += m_config.patch_tokens * cur_handle->visual_features.image_patch_counts[j];

                auto post_img_tokens = tokenizer->Encode(config.image_postfix);
                for (auto tid : post_img_tokens) input_ids.push_back(static_cast<int32_t>(tid));
            }

            std::string cur_req_post_prompt = cur_handle->generate_result.user_prompt + "<|im_end|>\n";
            for (auto tid : tokenizer->Encode(cur_req_post_prompt)) input_ids.push_back(static_cast<int32_t>(tid));

            std::string cur_req_output_prompt = "<|im_start|>assistant\n" + cur_handle->generate_result.outputs_text[0] + "<|im_end|>\n";
            for (auto tid : tokenizer->Encode(cur_req_output_prompt)) input_ids.push_back(static_cast<int32_t>(tid));
        }
        
        all_vis_feats.push_back(handle->visual_features);
        std::string cur_req_pre_prompt = "<|im_start|>user\n";
        for (auto tid : tokenizer->Encode(cur_req_pre_prompt)) input_ids.push_back(static_cast<int32_t>(tid));

        for (int j = 0; j < handle->visual_features.image_patch_counts.size(); ++j) {
            int width = (m_config.max_num_frames > 0) ? static_cast<int>(std::log10(m_config.max_num_frames)) + 1 : 1;
            std::string prefix = config.image_prefix;
            std::string placeholder = "$N$";

            std::ostringstream oss;
            oss << std::setw(width) << std::setfill('0') << (j + 1);
            std::string formatted_index = oss.str();

            size_t pos = prefix.find(placeholder);
            if (pos != std::string::npos) {
                prefix.replace(pos, placeholder.length(), formatted_index);
            }

            auto pre_img_tokens = tokenizer->Encode(prefix);
            for (auto tid : pre_img_tokens) input_ids.push_back(static_cast<int32_t>(tid));

            for (int k = 0; k < m_config.patch_tokens * handle->visual_features.image_patch_counts[j]; ++k) {
                input_ids.push_back(cur_fake_id + k);
            }
            cur_fake_id += m_config.patch_tokens * handle->visual_features.image_patch_counts[j];

            auto post_img_tokens = tokenizer->Encode(config.image_postfix);
            for (auto tid : post_img_tokens) input_ids.push_back(static_cast<int32_t>(tid));
        }

        std::string cur_req_post_prompt = handle->generate_result.user_prompt + "<|im_end|>\n";
        for (auto tid : tokenizer->Encode(cur_req_post_prompt)) input_ids.push_back(static_cast<int32_t>(tid));

        std::string cur_req_output_prompt = "<|im_start|>assistant\n";
        for (auto tid : tokenizer->Encode(cur_req_output_prompt)) input_ids.push_back(static_cast<int32_t>(tid));

    }

    // Create the Request object
    tle::Request request = create_request_from_dict_async(
        input_ids,
        all_vis_feats,
        m_config,
        config,
        metadata,
        m_stream,
        handle->generate_result.image_embeddings_gpu_ptr
    );

    std::uint64_t request_id = llm_executor->enqueueRequest(request);

    handle->generate_result.request_id = request_id;
    handle->generate_result.input_tokens = input_ids;
    handle->generate_result.outputs_tokens.resize(m_config.max_beam_width);
    handle->generate_result.last_outputs_token.resize(m_config.max_beam_width);
    handle->generate_result.start_gen = std::chrono::high_resolution_clock::now();

    if (config.streaming) {
        handle->generate_result.start_ttft = handle->generate_result.start_gen;
    }

    handle->generate_result.system_prompt = config.system_prompt;

}

void InternVL3LLMEngine::update_response(
    std::vector<SharedVisGenHandle>& handles,
    uint32_t timeout_ms = 1000,
    bool time_to_first_token_run = false
) {
    if (handles.size() == 0) return;
    auto responses = llm_executor->awaitResponses(std::chrono::milliseconds(timeout_ms));
    for (auto const& resp : responses) {
        std::uint64_t rid = resp.getRequestId();
        auto it = std::find_if(handles.begin(), handles.end(), 
            [rid, time_to_first_token_run](const SharedVisGenHandle handle) {
                if (time_to_first_token_run) return handle->generate_result.ttft_request_id == rid;
                else return handle->generate_result.request_id == rid;
            });
        if (it == handles.end()) continue;

        SharedVisGenHandle handle = *it;
        GenerateResult* res = &(handle->generate_result);
        
        if (resp.hasError()) {
            res->error_msg = resp.getErrorMsg();
            res->has_error = true;
        }

        if ((time_to_first_token_run) || 
           (handle->gen_config.profiling && 
            handle->gen_config.streaming && 
            !res->first_token_captured)
        ) {
            res->first_token_captured = true;
            res->end_ttft = std::chrono::high_resolution_clock::now();
            if (time_to_first_token_run) break;
        }

        auto const& result = resp.getResult();
        {
            std::lock_guard<std::mutex> lock(handle->data_mutex);
            if (handle->gen_config.streaming) {
                // Streaming, the llm_engine only return result of output tokens
                for (size_t beam_idx = 0; beam_idx < result.outputTokenIds.size(); ++beam_idx) {
                    auto const& tokens = result.outputTokenIds[beam_idx];
                    for (auto tokenId : tokens) {
                        res->outputs_tokens[beam_idx].push_back(static_cast<int>(tokenId));
                        res->last_outputs_token[beam_idx].push_back(static_cast<int>(tokenId));
                    }
                }
            } else {
                // Non-Streaming, the llm_engine return result of input + output tokens
                for (size_t beam_idx = 0; beam_idx < result.outputTokenIds.size(); ++beam_idx) {
                    auto const& tokens = result.outputTokenIds[beam_idx];
                    for (size_t i = res->input_tokens_len(); i < tokens.size(); ++i) {
                        res->outputs_tokens[beam_idx].push_back(static_cast<int>(tokens[i]));
                        res->last_outputs_token[beam_idx].push_back(static_cast<int>(tokens[i]));
                    }
                }
            }
            res->last_outputs_text = decode_outputs(
                res->last_outputs_token,
                0
            );
        }

        if (result.isFinal && !time_to_first_token_run) {
            res->done_output = true;
            res->outputs_text = decode_outputs(
                res->outputs_tokens,
                0
            );

            if (res->image_embeddings_gpu_ptr) {
                cudaFreeAsync(res->image_embeddings_gpu_ptr, m_stream);
                res->image_embeddings_gpu_ptr = nullptr;
            }       

            for (size_t m = 0; m < res->outputs_tokens.size(); ++m) {
                res->full_stops.push_back(res->outputs_tokens[m].back() == metadata.eos_id);
            }

            if (handle->gen_config.profiling) {
                res->end_gen = std::chrono::high_resolution_clock::now();
            }
        }

    }
}

} // namespace trt_multimodal