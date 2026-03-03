#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "TrtMultimodalRunner/IMultimodalRunner.hpp"

void print_gen_summary(const trt_multimodal::GenerateResult& res) {
    std::stringstream ss;

    const std::string CYAN = "\033[36m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m"; // 新增黄色用于文本
    const std::string BOLD = "\033[1m";
    const std::string RESET = "\033[0m";

    // --- 1. 性能指标部分 ---
    std::cout << "\n" << BOLD << CYAN << " 📊 Inference Performance Summary" << RESET << "\n";
    std::cout << " -------------------------------------------" << "\n";

    std::cout << "  " << std::left << std::setw(28) << "Input Tokens:" << res.input_tokens_len() << "\n";
    
    for (size_t i = 0; i < res.outputs_tokens_len().size(); ++i) {
        std::string label = "Output Tokens (Beam " + std::to_string(i) + "):";
        std::cout << "  " << std::left << std::setw(28) << label << res.outputs_tokens_len()[i] << "\n";
    }

    std::cout << " -------------------------------------------" << "\n";

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  " << std::left << std::setw(28) << "Total Latency:" 
              << GREEN << res.generation_latency_ms() << RESET << " ms\n";
    
    if (res.time_to_first_token_ms() > 0) {
        std::cout << "  " << std::left << std::setw(28) << "Time to First Token (TTFT):" 
                  << GREEN << res.time_to_first_token_ms() << RESET << " ms\n";
    }

    std::cout << " -------------------------------------------" << "\n";

    std::cout << "  " << std::left << std::setw(28) << "Tokens per Second (TPS):" 
              << BOLD << res.tokens_per_second() << RESET << " tok/s\n";
    ss << "  " << std::left << std::setw(28) << "System Throughput:" 
       << BOLD << res.system_throughput() << RESET << " tok/s\n";
    
    // --- 2. 模型输出文本部分 ---
    std::cout << "\n" << BOLD << YELLOW << " ✉️  Model Generated Outputs" << RESET << "\n";
    std::cout << " ===========================================" << "\n";

    if (res.outputs_text.empty()) {
        std::cout << "  " << "[No text generated]" << "\n";
    } else {
        for (size_t i = 0; i < res.outputs_text.size(); ++i) {
            // 如果是多 Beam 模式，打印 Beam 编号
            if (res.outputs_text.size() > 1) {
                std::cout << BOLD << "  Beam " << i << ":" << RESET << "\n";
            }
            
            // 打印实际生成的文本，带引号包裹
            std::cout << "  \"" << res.outputs_text[i] << "\"" << "\n";

            if (i < res.outputs_text.size() - 1) {
                std::cout << "  -------------------------------------------" << "\n";
            }
        }
    }
    std::cout << " ===========================================\n" << std::endl;
}

int main(int argc, char** argv) {

    std::string inputText = "Please describe the 2 images.";

    std::vector<std::string> imagePaths = {
        "/home/pi/kx/sample_images/dog.jpg",
        "/home/pi/kx/sample_images/cat.jpg"
    };

    trt_multimodal::ModelConfig m_config = trt_multimodal::ModelConfig();
    m_config.model_type = trt_multimodal::ModelType::Type::INTERNVL3;
    m_config.llm_engine_path = "/home/pi/kx/pt2engine_vlm/models/InternVL3-1B_i8/InternVL3-1B_llm_engine";
    m_config.vis_engine_path = "/home/pi/kx/pt2engine_vlm/models/InternVL3-1B_i8/InternVL3-1B_vis_engine/model.engine";
    m_config.tokenizer_path = "/home/pi/.cache/huggingface/hub/models--OpenGVLab--InternVL3-1B/snapshots/4415a3b810e636d11dfa86b0e9ba40bb00535aa8/tokenizer.json";

    m_config.max_beam_width = 1;
    m_config.max_llm_batch = 2;
    m_config.max_vis_batch = 2;
    m_config.patch_token_size = 256;
    m_config.embedding_dim = 896;

    trt_multimodal::GenerateConfig gen_config = trt_multimodal::GenerateConfig();
    gen_config.system_prompt = "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型, 英文名叫InternVL, 是一个有用无害的人工智能助手。";
    gen_config.image_prefix = "Image-$N$: ";
    gen_config.image_postfix = "\n";
    gen_config.max_new_tokens = 512;
    gen_config.top_k = 1;
    gen_config.top_p = 0.0f;
    gen_config.temperature = 0.2f;
    gen_config.repetition_penalty = 1.0f;
    gen_config.min_patch = 1;
    gen_config.max_patch = 1;
    gen_config.patch_size = 448;
    gen_config.use_thumbnail = false;
    gen_config.streaming = true;
    gen_config.profiling = true;

    std::unique_ptr<trt_multimodal::IMultimodalRunner> runner = trt_multimodal::IMultimodalRunner::create(m_config);

    std::vector<cv::Mat> frames;
    for (auto path : imagePaths){
        cv::Mat img = cv::imread(path);
        cv::Mat rgbImg;
        cv::cvtColor(img, rgbImg, cv::COLOR_BGR2RGB);

        frames.push_back(std::move(rgbImg));
    }

    trt_multimodal::VisualFeatures vis_feats;
    runner->extract_visual_features(
        frames,
        gen_config,
        vis_feats
    );

    trt_multimodal::GenerateResult gen_result;
    runner->generate_from_features(
        vis_feats,
        inputText,
        gen_config,
        gen_result
    );

    print_gen_summary(gen_result);

    return 0;
}
