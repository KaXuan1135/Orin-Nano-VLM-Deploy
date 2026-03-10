#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "TrtMultimodalRunner/IMultimodalRunner.hpp"

int main(int argc, char** argv) {

    std::string model_path =  "/home/pi/kx/pt2engine_vlm/models/InternVL3-1B_i8";

    trt_multimodal::ModelConfig m_config = trt_multimodal::ModelConfig();
    m_config.model_type = trt_multimodal::ModelType::Type::INTERNVL3;

    m_config.llm_engine_path = model_path + "/InternVL3-1B_llm_engine";
    m_config.vis_engine_path = model_path + "/InternVL3-1B_vis_engine/model.engine";
    m_config.tokenizer_path = model_path + "/tokenizers/tokenizer.json";

    m_config.max_beam_width = 1;
    m_config.max_llm_batch = 20;
    m_config.max_vis_batch = 6;
    m_config.patch_token_size = 256;
    m_config.embedding_dim = 896;

    trt_multimodal::GenerateConfig gen_config = trt_multimodal::GenerateConfig();
    gen_config.system_prompt = "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型, 英文名叫InternVL, 是一个有用无害的人工智能助手。";
    gen_config.image_prefix = "Image-$N$: ";
    gen_config.image_postfix = "\n";
    gen_config.max_new_tokens = 350;
    gen_config.top_k = 1;
    gen_config.top_p = 0.0f;
    gen_config.temperature = 0.2f;
    gen_config.repetition_penalty = 1.0f;
    gen_config.min_patch = 1;
    gen_config.max_patch = 1;
    gen_config.patch_size = 448;
    gen_config.use_thumbnail = false;
    gen_config.streaming = false;
    gen_config.profiling = true;

    int request_num = 20;

    std::string inputText = "Can you describe the 6 images?";

    std::vector<std::string> imagePaths = {
        "/home/pi/kx/sample_images/dog.jpg",
        "/home/pi/kx/sample_images/cat.jpg",
        "/home/pi/kx/sample_images/tiger.jpg",
        "/home/pi/kx/sample_images/wolf.jpg",
        "/home/pi/kx/sample_images/apple.jpg",
        "/home/pi/kx/sample_images/orange.jpg"
    };

    std::unique_ptr<trt_multimodal::IMultimodalRunner> runner = trt_multimodal::IMultimodalRunner::create(m_config);

    std::vector<cv::Mat> single_request_frames;
    for (const auto& path : imagePaths) {
        cv::Mat img = cv::imread(path);
        cv::Mat rgbImg;
        cv::cvtColor(img, rgbImg, cv::COLOR_BGR2RGB);
        single_request_frames.push_back(std::move(rgbImg));
    }

    std::vector<std::vector<cv::Mat>> batch_frames(request_num, single_request_frames);
    std::vector<std::string> batch_prompts(request_num, inputText);
    std::vector<trt_multimodal::GenerateConfig> batch_configs(request_num, gen_config);
    std::vector<trt_multimodal::GenerateResult> gen_results;

    auto all_gen_start = std::chrono::high_resolution_clock::now();

    runner->batch_generate(
        batch_frames,
        batch_prompts,
        batch_configs,
        gen_results
    );

    auto all_gen_end = std::chrono::high_resolution_clock::now();

    double total_elapsed_ms = std::chrono::duration<double, std::milli>(all_gen_end - all_gen_start).count();

    double total_tokens_generated = 0;
    double total_ttft_ms = 0;
    int valid_ttft_count = 0;

    for (int i = 0; i < request_num; ++i) {
        auto& res = gen_results[i];
        
        // 累加生成的 token 数量 (假设只取 beam 0)
        auto lens = res.outputs_tokens_len();
        if (!lens.empty()) {
            total_tokens_generated += lens[0];
        }
        
        // 累加 TTFT
        double ttft = res.time_to_first_token_ms();
        if (ttft > 0) {
            total_ttft_ms += ttft;
            valid_ttft_count++;
        }
    }

    double overall_tokens_per_second = (total_tokens_generated / total_elapsed_ms) * 1000.0;
    double avg_output_tokens = total_tokens_generated / request_num;
    double avg_ttft = (valid_ttft_count > 0) ? (total_ttft_ms / valid_ttft_count) : 0.0;

    std::cout << "\n" << std::string(40, '=') << std::endl;
    std::cout << "Performance Summary" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total Elapsed Time    : " << total_elapsed_ms << " ms" << std::endl;
    std::cout << "Overall Throughput    : " << overall_tokens_per_second << " tokens/s" << std::endl;
    std::cout << "Avg Output Tokens/Req : " << avg_output_tokens << " tokens" << std::endl;
    std::cout << "Average TTFT          : " << avg_ttft << " ms" << std::endl;
    std::cout << std::string(40, '=') << std::endl;


    auto batch_start = gen_results[0].start_gen; 
    auto batch_end = gen_results[0].end_gen;

    int32_t total_generated_tokens = 0;

    for (const auto& res : gen_results) {
        if (res.start_gen > batch_start) batch_start = res.start_gen;
        if (res.end_gen > batch_end) batch_end = res.end_gen;

        // Calculate generated tokens (Sequence length - prompt length)
        // outputs_tokens_len()[0] is the first beam
        int32_t gen_len = res.outputs_tokens_len()[0]; // - res.input_tokens_len();
        total_generated_tokens += std::max(0, gen_len);
    }

    std::chrono::duration<double> duration = batch_end - batch_start;
    double total_seconds = duration.count();

    double batch_tps = (total_seconds > 0) ? (total_generated_tokens / total_seconds) : 0.0;

    // std::cout << "\n--- Batch Generation Performance ---" << std::endl;
    // std::cout << "Total Tokens Generated : " << total_generated_tokens << " tokens" << std::endl;
    // std::cout << "Batch Wall-clock Time  : " << std::fixed << std::setprecision(4) << total_seconds << " s" << std::endl;
    // std::cout << "Batch Throughput       : " << std::fixed << std::setprecision(2) << batch_tps << " tokens/sec" << std::endl;
    // std::cout << "------------------------------------" << std::endl;
    // std::cout << "\n" << "Model Generated Outputs (Batch 0)" << "\n";
    // std::cout << " ===========================================" << "\n";
    // if (gen_results[0].outputs_text.empty()) {
    //     std::cout << "  " << "[No text generated]" << "\n";
    // } else {
    //     for (size_t i = 0; i < gen_results[0].outputs_text.size(); ++i) {
    //         if (gen_results[0].outputs_text.size() > 1) {
    //             std::cout << "  Beam " << i << ":" << "\n";
    //         }
    //         std::cout << "  \"" << gen_results[0].outputs_text[i] << "\"" << "\n";
    //         if (i < gen_results[0].outputs_text.size() - 1) {
    //             std::cout << "  -------------------------------------------" << "\n";
    //         }
    //     }
    // }
    // std::cout << " ===========================================\n" << std::endl;

    return 0;
}
