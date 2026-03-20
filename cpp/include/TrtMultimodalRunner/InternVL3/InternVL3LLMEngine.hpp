#pragma once
#include <cuda_runtime.h>

#include "tokenizers_cpp.h"
#include "tensorrt_llm/executor/executor.h"
#include "TrtMultimodalRunner/Types.hpp" 

namespace trt_multimodal {

class InternVL3LLMEngine {
public:

    InternVL3LLMEngine(
        const ModelConfig& config,
        const cudaStream_t& stream
    );

    void generate_from_features(
        std::vector<SharedVisGenHandle> handles
    );

    void enqueue_generate_from_features(
        SharedVisGenHandle& handle
    );

    void update_response(
        std::vector<SharedVisGenHandle>& handles,
        uint32_t timeout_ms,
        bool time_to_first_token_run
    );

private:

    ModelConfig m_config;
    cudaStream_t m_stream;
    TokenizerMetadata metadata;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    std::unique_ptr<tensorrt_llm::executor::Executor> llm_executor;

    std::vector<std::string> decode_outputs(
        const std::vector<std::vector<int32_t>> beams_tokens,
        size_t input_len
    );

};

} // namespace trt_multimodal