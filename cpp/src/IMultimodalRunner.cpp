#include <iostream>

#include "TrtMultimodalRunner/IMultimodalRunner.hpp"
#include "TrtMultimodalRunner/InternVL3/InternVL3Runner.hpp"

namespace trt_multimodal {

std::unique_ptr<IMultimodalRunner> IMultimodalRunner::create( 
    const ModelConfig& model_config
) {
    try {
        if (model_config.model_type == ModelType::Type::INTERNVL3) return std::make_unique<InternVL3Runner>(model_config);
        //else if other model
    } catch (const std::exception& e) {
        std::cerr << "[IMultimodalRunner] Failed to initialize " << ModelType::to_string(model_config.model_type) << ": " << e.what() << std::endl;
        return nullptr;
    }

    std::cerr << "[IMultimodalRunner] Unsupported model type: '" << ModelType::to_string(model_config.model_type) << "'. "
              << "Currently supported types are: [" << ModelType::get_supported_model_types() << "]." << std::endl;

    return nullptr;
}

}