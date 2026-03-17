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
    const VisualFeatures& vis_features,
    const ModelConfig& m_config,
    const GenerateConfig& gen_config,
    const TokenizerMetadata& metadata
) {
    
    tle::SamplingConfig sampling_config(m_config.max_beam_width);
    sampling_config.setTopK(gen_config.top_k);
    sampling_config.setTopP(gen_config.top_p ? gen_config.top_p > 0.0f : 1.0f); 
    sampling_config.setTemperature(gen_config.temperature);
    sampling_config.setRepetitionPenalty(gen_config.repetition_penalty);

    tle::Tensor embedding = tle::Tensor::of(
        tle::DataType::kBF16, // TODO: make it depends on vis_features.dtype
        vis_features.embeddings_ptr.get(),
        tle::Shape{
            static_cast<long int>(vis_features.total_patches() * m_config.patch_token_size), 
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

    llm_executor = std::make_unique<tensorrt_llm::executor::Executor>(
        m_config.llm_engine_path, 
        tensorrt_llm::executor::ModelType::kDECODER_ONLY,
        tensorrt_llm::executor::ExecutorConfig(m_config.max_beam_width)
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
    const std::vector<VisualFeatures>& vis_features,
    const std::vector<std::string>& user_prompts, // Changed to plural for clarity
    const std::vector<GenerateConfig>& gen_configs,
    std::vector<GenerateResult>& gen_results
) {
    size_t batch_size = vis_features.size();
    gen_results.clear();
    gen_results.reserve(batch_size);

    std::vector<tle::Request> requests;
    requests.reserve(batch_size);

    // 1. Construct Inputs and Requests for the whole batch
    for (size_t b = 0; b < batch_size; ++b) {
        const auto& config = gen_configs[b];
        
        // Construct Textual Prompts
        std::string pre_prompt = "<|im_start|>system\n" + config.system_prompt + "<|im_end|>\n<|im_start|>user\n";
        std::string post_prompt = "\n" + user_prompts[b] + "<|im_end|>\n<|im_start|>assistant\n";

        std::vector<int32_t> input_ids;
        size_t cur_fake_id = tokenizer->GetVocabSize();

        // Encode System/User Prefix
        auto pre_tokens = tokenizer->Encode(pre_prompt);
        for (auto tid : pre_tokens) input_ids.push_back(static_cast<int32_t>(tid));

        // Encode Image Patches with Placeholders
        int num_patches = vis_features[b].total_patches(); 
        for (int i = 0; i < num_patches; ++i) {
            std::string prefix = config.image_prefix;
            std::string placeholder = "$N$";
            size_t pos = prefix.find(placeholder);
            if (pos != std::string::npos) {
                prefix.replace(pos, placeholder.length(), std::to_string(i + 1));
            }

            auto pre_img_tokens = tokenizer->Encode(prefix);
            for (auto tid : pre_img_tokens) input_ids.push_back(static_cast<int32_t>(tid));

            // Insert Fake IDs for Visual Embeddings
            for (int j = 0; j < m_config.patch_token_size; ++j) {
                input_ids.push_back(cur_fake_id + j);
            }
            cur_fake_id += m_config.patch_token_size;

            auto post_img_tokens = tokenizer->Encode(config.image_postfix);
            for (auto tid : post_img_tokens) input_ids.push_back(static_cast<int32_t>(tid));
        }

        // Encode User Message Suffix
        auto post_tokens = tokenizer->Encode(post_prompt);
        for (auto tid : post_tokens) input_ids.push_back(static_cast<int32_t>(tid));

        // Create the Request object
        requests.push_back(create_request_from_dict(
            input_ids,
            vis_features[b],
            m_config,
            config,
            metadata
        ));

        gen_results.emplace_back(); 
        GenerateResult& res = gen_results.back();

        // Now populate the reference 'res'
        res.gen_config = config;
        res.user_prompt = user_prompts[b];
        res.input_tokens = input_ids;
        res.outputs_tokens.resize(m_config.max_beam_width);
        res.last_outputs_token.resize(m_config.max_beam_width);
        res.start_gen = std::chrono::high_resolution_clock::now();
        
        if (config.streaming) {
            res.start_ttft = res.start_gen;
        }
    }

    // 2. Enqueue all requests to the LLM Executor
    for (size_t b = 0; b < batch_size; ++b) {
        gen_results[b].request_id = llm_executor->enqueueRequest(requests[b]);
    }

    // 3. Multi-request polling loop
    bool all_done = false;
    while (!all_done) {
        all_done = true;
        std::vector<GenerateResult*> active_results;
        
        for (size_t b = 0; b < batch_size; ++b) {
            if (!gen_results[b].done_output) {
                active_results.push_back(&gen_results[b]);
                all_done = false; 
            }
        }
        if (!active_results.empty()) {
            update_response(active_results, 1000, false);
            
            // Check for errors in the batch
            for (auto* res : active_results) {
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
        if (gen_configs[b].profiling && !gen_configs[b].streaming) {
            GenerateConfig ttft_config = gen_configs[b];
            ttft_config.max_new_tokens = 1;

            tle::Request ttft_req = create_request_from_dict(
                gen_results[b].input_tokens,
                vis_features[b],
                m_config,
                ttft_config,
                metadata
            );

            gen_results[b].start_ttft = std::chrono::high_resolution_clock::now();
            gen_results[b].ttft_request_id = llm_executor->enqueueRequest(ttft_req);
            // Reset capture flag for the profiling run
            gen_results[b].first_token_captured = false; 

            while (!gen_results[b].first_token_captured) {
                std::vector<GenerateResult*> ptr_vec = {&gen_results[b]};
                update_response(ptr_vec, 1000, true);
            }
        }
    }
}

void InternVL3LLMEngine::enqueue_generate_from_features(
    const VisualFeatures& vis_features,
    const std::string& user_prompt,
    const GenerateConfig& gen_config,
    SharedVisGenHandle& handle
) {

    if (handle->prev_handles.size() > 0) {
        std::cout << "Has prev" << std::endl;
    } else {
        std::cout << "No prev" << std::endl;
    }

    std::string pre_prompt;
    if (handle->prev_handles.size() > 0) {
        pre_prompt = "<|im_end|>\n<|im_start|>user\n";
    } else {
        pre_prompt = "<|im_start|>system\n" + gen_config.system_prompt + "<|im_end|>\n<|im_start|>user\n";
    }
    std::string post_prompt = "\n" + user_prompt + "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<std::string> images_prefix(vis_features.total_patches());
    std::vector<std::string> images_postfix(vis_features.total_patches());

    for (int i = 0; i < vis_features.total_patches(); ++i) {
        std::string prefix = gen_config.image_prefix;
        std::string placeholder = "$N$";
        std::string replacement = std::to_string(i + 1);
        
        size_t pos = prefix.find(placeholder);
        if (pos != std::string::npos) {
            prefix.replace(pos, placeholder.length(), replacement);
        }
        images_prefix[i] = prefix;
        images_postfix[i] = gen_config.image_postfix;
    }

    //Constuct Input Ids, Replace Image Token with fake tokens
    std::vector<int32_t> input_ids;
    size_t cur_fake_id = tokenizer->GetVocabSize();
    std::cout << "If consider only this chat, the vocab size is " << cur_fake_id << std::endl;

    if (handle->prev_handles.size() > 0) {
        for (auto const& prev_handle : handle->prev_handles) {
            for (auto const& tid : prev_handle->generate_result.input_tokens) {
                cur_fake_id = std::max(static_cast<long>(cur_fake_id), static_cast<long>(tid));
            }
        }
        std::cout << "If consider previous chats, the vocab size is " << cur_fake_id << std::endl;
    }



    if (handle->prev_handles.size() == 0) {
        std::vector<int32_t> pre_prompt_tokens = tokenizer->Encode(pre_prompt);
        for (auto tid : pre_prompt_tokens) input_ids.push_back(static_cast<int32_t>(tid));

        for (int i = 0; i < vis_features.image_patch_counts.size(); ++i) {
            std::vector<int32_t> pre_img_tokens = tokenizer->Encode(images_prefix[i]);
            for (auto tid : pre_img_tokens) input_ids.push_back(static_cast<int32_t>(tid));

            for (int j = 0; j < m_config.patch_token_size; ++j) input_ids.push_back(cur_fake_id + j);
            cur_fake_id += m_config.patch_token_size;

            std::vector<int32_t> post_img_tokens = tokenizer->Encode(images_postfix[i]);
            for (auto tid : post_img_tokens) input_ids.push_back(static_cast<int32_t>(tid));
        }
        
        std::vector<int32_t> post_prompt_tokens = tokenizer->Encode(post_prompt);
        for (auto tid : post_prompt_tokens) input_ids.push_back(static_cast<int32_t>(tid));
    } else {

        size_t count = 0;
        std::vector<int32_t> pre_prompt_tokens = tokenizer->Encode(pre_prompt);
        count += pre_prompt_tokens.size();

        for (int i = 0; i < vis_features.image_patch_counts.size(); ++i) {
            std::vector<int32_t> pre_img_tokens = tokenizer->Encode(images_prefix[i]);
            count += pre_prompt_tokens.size();

            // for (int j = 0; j < m_config.patch_token_size; ++j) input_ids.push_back(cur_fake_id + j);
            count += m_config.patch_token_size;

            std::vector<int32_t> post_img_tokens = tokenizer->Encode(images_postfix[i]);
            count += post_img_tokens.size();
        }
        
        std::vector<int32_t> post_prompt_tokens = tokenizer->Encode(post_prompt);
        count += post_prompt_tokens.size();

        // size_t max_input_len = 4036;
        // size_t max_input_len = 1000;
        std::cout << "max_input_len is hardcoded to " << m_config.max_input_len << std::endl;
        std::cout << "current input_len is " << count << std::endl;
 
        size_t history_count = 0;
        auto const& prev_handle = handle->prev_handles[handle->prev_handles.size() - 1];
        
        history_count += prev_handle->generate_result.input_tokens.size();
        history_count += prev_handle->generate_result.outputs_tokens[0].size();

        std::cout << "history input_len is " << history_count << std::endl;

        if (count + history_count > m_config.max_input_len) {

            std::cout << "[Dynamic Window] Enabled. Limit: " << m_config.max_input_len 
                        << ", Current Prompt: " << count << std::endl;
            std::vector<int32_t> system_prompt_token = tokenizer->Encode("<|im_start|>system\n" + gen_config.system_prompt + "<|im_end|>\n");

            size_t cur_count = count + system_prompt_token.size();

            std::vector<std::vector<int32_t>> latest_user_prompt_token;

            size_t total_history_chats = handle->prev_handles.size();
            size_t retained_chats = 0;

            for (int i = handle->prev_handles.size() - 1; i >= 0; --i) {
                auto const& cur_handle = handle->prev_handles[i];
                std::string cur_req_prompt = "<|im_start|>user\n" + cur_handle->generate_result.user_prompt + "<|im_end|>\n";
                cur_req_prompt += "<|im_start|>assistant\n" + cur_handle->generate_result.outputs_text[0] + "<|im_end|>\n";

                std::vector<int32_t> cur_req_prompt_token = tokenizer->Encode(cur_req_prompt);
                
                if (cur_count + cur_req_prompt_token.size() > m_config.max_input_len) { 

                    std::cout << "[Dynamic Window] Limit reached at chat index " << i 
                      << ". Dropping oldest " << (i + 1) << " chats." << std::endl;

                    break;
                } else {
                    latest_user_prompt_token.push_back(cur_req_prompt_token);
                    cur_count += cur_req_prompt_token.size();
                    retained_chats++;
                }
            }

            std::cout << "[Dynamic Window] Summary: Total Chats: " << total_history_chats 
              << ", Retained: " << retained_chats 
              << ", Final History Input Tokens: " << cur_count << std::endl;

            for (auto const& tid : system_prompt_token) {
                input_ids.push_back(static_cast<int32_t>(tid));
            }

            for (int i = latest_user_prompt_token.size() - 1; i >= 0; --i) {
                for (auto const& tid : latest_user_prompt_token[i]) {
                    input_ids.push_back(static_cast<int32_t>(tid));
                }
            }

            std::vector<int32_t> pre_prompt_tokens = tokenizer->Encode(pre_prompt);
            for (auto tid : pre_prompt_tokens) input_ids.push_back(static_cast<int32_t>(tid));

            for (int i = 0; i < vis_features.image_patch_counts.size(); ++i) {
                std::vector<int32_t> pre_img_tokens = tokenizer->Encode(images_prefix[i]);
                for (auto tid : pre_img_tokens) input_ids.push_back(static_cast<int32_t>(tid));

                for (int j = 0; j < m_config.patch_token_size; ++j) input_ids.push_back(cur_fake_id + j);
                cur_fake_id += m_config.patch_token_size;

                std::vector<int32_t> post_img_tokens = tokenizer->Encode(images_postfix[i]);
                for (auto tid : post_img_tokens) input_ids.push_back(static_cast<int32_t>(tid));
            }
            
            std::vector<int32_t> post_prompt_tokens = tokenizer->Encode(post_prompt);
            for (auto tid : post_prompt_tokens) input_ids.push_back(static_cast<int32_t>(tid));

        } else {
            for (auto const& tid : prev_handle->generate_result.input_tokens) {
                input_ids.push_back(static_cast<int32_t>(tid));
            }
            for (auto const& tid: prev_handle->generate_result.outputs_tokens[0]) { // consider only beams 0 now
                input_ids.push_back(static_cast<int32_t>(tid));
            }
            std::cout << input_ids.size() << " of history being inherited" << std::endl;

            std::vector<int32_t> pre_prompt_tokens = tokenizer->Encode(pre_prompt);
            for (auto tid : pre_prompt_tokens) input_ids.push_back(static_cast<int32_t>(tid));

            for (int i = 0; i < vis_features.image_patch_counts.size(); ++i) {
                std::vector<int32_t> pre_img_tokens = tokenizer->Encode(images_prefix[i]);
                for (auto tid : pre_img_tokens) input_ids.push_back(static_cast<int32_t>(tid));

                for (int j = 0; j < m_config.patch_token_size; ++j) input_ids.push_back(cur_fake_id + j);
                cur_fake_id += m_config.patch_token_size;

                std::vector<int32_t> post_img_tokens = tokenizer->Encode(images_postfix[i]);
                for (auto tid : post_img_tokens) input_ids.push_back(static_cast<int32_t>(tid));
            }
            
            std::vector<int32_t> post_prompt_tokens = tokenizer->Encode(post_prompt);
            for (auto tid : post_prompt_tokens) input_ids.push_back(static_cast<int32_t>(tid));

        }
    }

    tle::Request request = create_request_from_dict(
        input_ids,
        vis_features,
        m_config,
        gen_config,
        metadata
    );

    if (gen_config.profiling) {
        handle->generate_result.start_gen = std::chrono::high_resolution_clock::now();
        if (gen_config.streaming) {
            handle->generate_result.start_ttft = std::chrono::high_resolution_clock::now();
        }
    }

    std::uint64_t request_id = llm_executor->enqueueRequest(request);
    handle->generate_result.gen_config = gen_config;
    handle->generate_result.request_id = request_id;
    handle->generate_result.system_prompt = gen_config.system_prompt;
    handle->generate_result.user_prompt = user_prompt;
    handle->generate_result.input_tokens = input_ids;
    handle->generate_result.start_gen = std::chrono::high_resolution_clock::now();
    handle->generate_result.outputs_tokens.resize(m_config.max_beam_width);
    handle->generate_result.last_outputs_token.resize(m_config.max_beam_width);

}

void InternVL3LLMEngine::update_response(
    std::vector<GenerateResult*>& request_results,
    uint32_t timeout_ms = 1000,
    bool time_to_first_token_run = false
) {
    auto responses = llm_executor->awaitResponses(std::chrono::milliseconds(timeout_ms));
    for (auto const& resp : responses) {
        std::uint64_t rid = resp.getRequestId();
        auto it = std::find_if(request_results.begin(), request_results.end(), 
            [rid, time_to_first_token_run](const GenerateResult* res) {
                if (!res) return false;
                if (time_to_first_token_run) return res->ttft_request_id == rid;
                else return res->request_id == rid;
            });
        if (it == request_results.end()) continue;
        
        GenerateResult* res = *it;
        
        if (resp.hasError()) {
            res->error_msg = resp.getErrorMsg();
            res->has_error = true;
        }
        if ((time_to_first_token_run) || 
           (res->gen_config.profiling && 
            res->gen_config.streaming && 
            !res->first_token_captured)
        ) {
            res->first_token_captured = true;
            res->end_ttft = std::chrono::high_resolution_clock::now();
            if (time_to_first_token_run) break;
        }
        auto const& result = resp.getResult();

        {
            std::lock_guard<std::mutex> lock(res->data_mutex);

            if (res->gen_config.streaming) {
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

            for (size_t m = 0; m < res->outputs_tokens.size(); ++m) {
                res->full_stops.push_back(res->outputs_tokens[m].back() == metadata.eos_id);
            }

            if (res->gen_config.profiling) {
                res->end_gen = std::chrono::high_resolution_clock::now();
            }
        }

    }
}

} // namespace trt_multimodal