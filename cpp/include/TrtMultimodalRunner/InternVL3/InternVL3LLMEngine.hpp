#include <cuda_runtime.h>

#include "tokenizers_cpp.h"
#include "tensorrt_llm/executor/executor.h"
#include "TrtMultimodalRunner/Types.hpp" 

namespace trt_multimodal {

class InternVL3LLMEngine {
public:

    int init(
        const ModelConfig& config,
        const cudaStream_t& stream
    );    

    GenerateResult generate_from_features(
        const VisualFeatures& visual_features,
        const std::string& user_prompt,
        const GenerateConfig& generate_config
    );

private:

    ModelConfig m_config;
    cudaStream_t m_stream;

    std::unique_ptr<tensorrt_llm::executor::Executor> llm_executor;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    TokenizerMetadata metadata;

    std::vector<std::string> decode_outputs_cpp(
        const std::vector<int32_t>& output_ids, // 扁平化的 ids
        std::vector<size_t> total_seq_len,
        size_t input_len
    );
};

} // namespace trt_multimodal