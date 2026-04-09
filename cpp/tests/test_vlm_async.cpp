#include <vector>
#include <string>
#include <thread>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "TrtMultimodalRunner/Types.hpp"
#include "TrtMultimodalRunner/IAsyncMultimodalRunner.hpp"

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
              << GREEN << res.generation_latency() << RESET << " s\n";
    
    if (res.time_to_first_token() > 0) {
        std::cout << "  " << std::left << std::setw(28) << "Time to First Token (TTFT):" 
                  << GREEN << res.time_to_first_token() << RESET << " s\n";
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

    std::string inputText = "Can you describe the 6 images?";

    std::vector<std::string> imagePaths = {
        "/home/pi/kx/sample_images/cat.jpg", 
        "/home/pi/kx/sample_images/tiger.jpg",
        "/home/pi/kx/sample_images/apple.jpg",
        "/home/pi/kx/sample_images/orange.jpg",
        "/home/pi/kx/sample_images/airplane.jpg",
        "/home/pi/kx/sample_images/car.jpg"
    };

    trt_multimodal::ModelConfig m_config = trt_multimodal::ModelConfig(
        trt_multimodal::ModelType::Type::INTERNVL3,
        "/mnt/sdcard/models/InternVL3-1B_i8"
    );

    trt_multimodal::GenerateConfig gen_config = trt_multimodal::GenerateConfig();
    gen_config.system_prompt = "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型, 英文名叫InternVL, 是一个有用无害的人工智能助手。";
    gen_config.image_prefix = "Image-$N$: ";
    gen_config.image_postfix = "\n";
    gen_config.max_new_tokens = 512;
    gen_config.top_k = 1;
    gen_config.top_p = 0.0f;
    gen_config.temperature = 1.0f;
    gen_config.repetition_penalty = 1.0f;
    gen_config.min_patch = 1;
    gen_config.max_patch = 1;
    // gen_config.patch_size = 448;
    gen_config.use_thumbnail = false;
    gen_config.streaming = true;
    gen_config.profiling = true;

    int request_num = 20;

    std::unique_ptr<trt_multimodal::IAsyncMultimodalRunner> runner = trt_multimodal::IAsyncMultimodalRunner::create(m_config);

    std::vector<cv::Mat> frames;
    for (auto path : imagePaths){
        cv::Mat img = cv::imread(path);
        cv::Mat rgbImg;
        cv::cvtColor(img, rgbImg, cv::COLOR_BGR2RGB);

        frames.push_back(std::move(rgbImg));
    }

    std::vector<trt_multimodal::SharedVisGenHandle> vis_handles;
    for (int i = 0; i < request_num; ++i) {
        trt_multimodal::SharedVisGenHandle handle = runner->create_handle(
            gen_config,
            inputText,
            frames
        );
        runner->enqueue_extract_visual_features(handle);
        vis_handles.push_back(handle);
    }

    // --- 阶段 2: 统一等待视觉提取完成 ---
    // 只要还有一个没完成，就继续等
    // 真正异步推理应该是只要 vis 结束就可以交给 LLM
    bool all_vis_done = false;
    while (!all_vis_done) {
        all_vis_done = true;
        for (auto& v_h : vis_handles) {
            if (!v_h->vis_finished.load()) {
                all_vis_done = false;
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    auto all_gen_start = std::chrono::high_resolution_clock::now();

    std::vector<trt_multimodal::SharedVisGenHandle> gen_handles;
    for (int i = 0; i < request_num; ++i) {
        runner->enqueue_generate_from_features(vis_handles[i]);
        gen_handles.push_back(vis_handles[i]);
    }

    while (true) {
        bool all_gen_done = true;
        for (size_t i = 0; i < gen_handles.size(); ++i) {
            if (!gen_handles[i]->gen_finished.load()){
                all_gen_done = false;
            }
            // auto& res = gen_handles[i]->generate_result;
            // if (!res.done_output.load()) {
            //     all_gen_done = false;
            // }
        }
        if (all_gen_done) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    auto all_gen_end = std::chrono::high_resolution_clock::now();

    double total_elapsed_ms = std::chrono::duration<double, std::milli>(all_gen_end - all_gen_start).count();

    double total_tokens_generated = 0;
    double total_ttft_ms = 0;
    int valid_ttft_count = 0;

    for (int i = 0; i < request_num; ++i) {
        auto& res = gen_handles[i]->generate_result;
        
        // 累加生成的 token 数量 (假设只取 beam 0)
        if (gen_config.streaming) {
            auto lens = res.outputs_tokens_len();
            if (!lens.empty()) {
                total_tokens_generated += lens[0];
            }
        } else{
            auto lens = res.outputs_tokens_len()[0] - res.input_tokens_len();
            total_tokens_generated += lens;
        }
        
        // 累加 TTFT
        double ttft = res.time_to_first_token();
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


    // for (int i = 0; i < request_num; ++i) {
    //     print_gen_summary(gen_handles[i]->generate_result);
    // }

    return 0;

}
