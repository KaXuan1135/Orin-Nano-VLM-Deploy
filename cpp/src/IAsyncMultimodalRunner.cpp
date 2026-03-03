#include <iostream>

#include "TrtMultimodalRunner/IAsyncMultimodalRunner.hpp"
#include "TrtMultimodalRunner/InternVL3/AsyncInternVL3Runner.hpp"

namespace trt_multimodal {

std::unique_ptr<IAsyncMultimodalRunner> IAsyncMultimodalRunner::create( 
    const ModelConfig& model_config
) {
    try {
        if (model_config.model_type == ModelType::Type::INTERNVL3) return std::make_unique<AsyncInternVL3Runner>(model_config);
        //else if other model
    } catch (const std::exception& e) {
        std::cerr << "[IAsyncMultimodalRunner] Failed to initialize " << ModelType::to_string(model_config.model_type) << ": " << e.what() << std::endl;
        return nullptr;
    }

    std::cerr << "[IAsyncMultimodalRunner] Unsupported model type: '" << ModelType::to_string(model_config.model_type) << "'. "
              << "Currently supported types are: [" << ModelType::get_supported_model_types() << "]." << std::endl;

    return nullptr;
}

}