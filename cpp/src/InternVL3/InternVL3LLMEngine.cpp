#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <fstream>
#include <cassert>
#include <iostream>
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

TokenizerMetadata get_tokenizer_ids(const std::string& path) {
    std::ifstream f(path);
    json data = json::parse(f);
    TokenizerMetadata meta;

    // 优先从 added_tokens 找 (这些通常具有最高优先级)
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

int InternVL3LLMEngine::init(
    const ModelConfig& config,
    const cudaStream_t& stream
) {

    m_config = config;
    m_stream = stream;

    initTrtLlmPlugins();

    llm_executor = std::make_unique<tensorrt_llm::executor::Executor>(
        config.llm_engine_path, 
        tensorrt_llm::executor::ModelType::kDECODER_ONLY,
        tensorrt_llm::executor::ExecutorConfig(config.max_beam_width)
    );

    tokenizer = tokenizers::Tokenizer::FromBlobJSON(read_file(config.tokenizer_path));
    metadata = get_tokenizer_ids(config.tokenizer_path);

    return 0;
}

std::string strip(const std::string& s) {
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start)) start++;
    auto end = s.end();
    do { end--; } while (std::distance(start, end) > 0 && std::isspace(*end));
    return std::string(start, end + 1);
}

std::vector<std::string> InternVL3LLMEngine::decode_outputs(
    // const std::vector<int32_t>& flat_beams_tokens, // 扁平化的 ids
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
            results[m] = strip(tokenizer->Decode(output_ids));
        }
    }

    return results;
}

void InternVL3LLMEngine::generate_from_features(
    const VisualFeatures& vis_features,
    const std::string& user_prompt,
    const GenerateConfig& gen_config,
    GenerateResult& gen_result
) {

    //Construct Input Prompts
    std::string pre_prompt = "<|im_start|>system\n" + gen_config.system_prompt + "<|im_end|>\n<|im_start|>user\n";
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
        images_prefix.push_back(prefix);
        images_postfix.push_back(gen_config.image_postfix);
    }

    //Constuct Input Ids, Replace Image Token with fake tokens
    std::vector<int32_t> input_ids;
    size_t cur_fake_id = tokenizer->GetVocabSize();

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
    
    tle::Request request = create_request_from_dict(
        input_ids,
        vis_features,
        m_config,
        gen_config,
        metadata
    );

    if (gen_config.profiling) {
        gen_result.start_gen = std::chrono::high_resolution_clock::now();
        if (gen_config.streaming) {
            gen_result.start_ttft = std::chrono::high_resolution_clock::now();
        }
    }

    std::uint64_t request_id = llm_executor->enqueueRequest(request);
    gen_result.gen_config = gen_config;
    gen_result.request_id = request_id;
    gen_result.system_prompt = gen_config.system_prompt;
    gen_result.user_prompt = user_prompt;
    gen_result.input_tokens = input_ids;
    gen_result.start_gen = std::chrono::high_resolution_clock::now();
    gen_result.outputs_tokens.resize(m_config.max_beam_width);
    gen_result.last_outputs_token.resize(m_config.max_beam_width);

    while(!gen_result.done_output) {
        std::vector<GenerateResult*> ptr_vec = {&gen_result};
        update_response(ptr_vec, 1000, false);
        if (gen_result.has_error) throw std::runtime_error("TRT-LLM Error for Request " + std::to_string(gen_result.request_id) + ": " + gen_result.error_msg);
    }

    if (gen_config.profiling && !gen_config.streaming) {

        std::cout << "User requested profiling, and not streaming, running generate one more time to estimate time to first token" << std::endl;

        GenerateConfig ttft_config = gen_config;
        ttft_config.max_new_tokens = 1;

        // 这里再跑一次 llm executor，复用之前的 visual encoded embeddings
        tle::Request ttft_request = create_request_from_dict(
            input_ids,
            vis_features,
            m_config,
            ttft_config,
            metadata
        );

        gen_result.start_ttft = std::chrono::high_resolution_clock::now();
        std::uint64_t ttft_request_id = llm_executor->enqueueRequest(ttft_request);

        while(!gen_result.first_token_captured) {
            std::vector<GenerateResult*> ptr_vec = {&gen_result};
            update_response(ptr_vec, 1000, true);
            if (gen_result.has_error) throw std::runtime_error("TRT-LLM Error for Request " + std::to_string(gen_result.ttft_request_id) + ": " + gen_result.error_msg);
        }

    }

}

SharedGenHandle InternVL3LLMEngine::enqueue_generate_from_features(
    const VisualFeatures& vis_features,
    const std::string& user_prompt,
    const GenerateConfig& gen_config
) {

    auto handle = std::make_shared<GenHandle>();

    //Construct Input Prompts
    std::string pre_prompt = "<|im_start|>system\n" + gen_config.system_prompt + "<|im_end|>\n<|im_start|>user\n";
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
        images_prefix.push_back(prefix);
        images_postfix.push_back(gen_config.image_postfix);
    }

    //Constuct Input Ids, Replace Image Token with fake tokens
    std::vector<int32_t> input_ids;
    size_t cur_fake_id = tokenizer->GetVocabSize();

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

    return handle;

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
            res->has_error.store(true);
        }
        if ((time_to_first_token_run) || 
           (res->gen_config.profiling && 
            res->gen_config.streaming && 
            !res->first_token_captured)
        ) {
            res->first_token_captured.store(true);
            res->end_ttft = std::chrono::high_resolution_clock::now();
        }
        auto const& result = resp.getResult();
        for (size_t beam_idx = 0; beam_idx < result.outputTokenIds.size(); ++beam_idx) {
            auto const& tokens = result.outputTokenIds[beam_idx];
            for (auto tokenId : tokens) {
                res->outputs_tokens[beam_idx].push_back(static_cast<int>(tokenId));
            }
            res->last_outputs_token[beam_idx] = tokens;
        }

        res->last_outputs_text = decode_outputs(
            res->last_outputs_token,
            res->gen_config.streaming ? 0 : res->input_tokens_len()
        );

        if (result.isFinal && !time_to_first_token_run) {
            res->done_output.store(true);
            res->outputs_text = decode_outputs(
                res->outputs_tokens,
                res->gen_config.streaming ? 0 : res->input_tokens_len()
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