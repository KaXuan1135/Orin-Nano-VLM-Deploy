#include <ctime>
#include <vector>
#include <string>
#include <thread>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "TrtMultimodalRunner/Types.hpp"
#include "TrtMultimodalRunner/IAsyncMultimodalRunner.hpp"

using namespace std::chrono_literals;

int main(int argc, char** argv) {

    std::srand(std::time(nullptr));
    
    int request_num = 2; // 100

    // 64 + > 1 (2)
    std::string inputText = "Hi can you please answer on what is the main activity happening in these frames?";
    // std::string inputText = "Can you please answer on what is the main activity happening in these frames?";


    trt_multimodal::ModelConfig m_config = trt_multimodal::ModelConfig(
        trt_multimodal::ModelType::Type::INTERNVL3,
        "/mnt/sdcard/models/InternVL3-1B_fp16_w(i8)a(bf16)_kv(bf16)"
        // "/mnt/sdcard/models/InternVL3-1B_fp16_w(i4)a(bf16)_kv(bf16)"
    );

    trt_multimodal::GenerateConfig gen_config = trt_multimodal::GenerateConfig();
    gen_config.system_prompt = "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型, 英文名叫InternVL, 是一个有用无害的人工智能助手。";
    gen_config.image_prefix = "Image-$N$: ";
    gen_config.image_postfix = "\n";
    gen_config.max_new_tokens = 1;
    gen_config.top_k = 1;
    gen_config.top_p = 0.0f;
    gen_config.temperature = 0.2f;
    gen_config.repetition_penalty = 1.0f;
    gen_config.min_patch = 1;
    gen_config.max_patch = 1;
    gen_config.use_thumbnail = false;
    gen_config.streaming = true;
    gen_config.profiling = true;

    std::unique_ptr<trt_multimodal::IAsyncMultimodalRunner> runner = trt_multimodal::IAsyncMultimodalRunner::create(m_config);

    auto overall_start = std::chrono::high_resolution_clock::now();

    std::vector<trt_multimodal::SharedVisGenHandle> aio_handles;
    for (int i = 0; i < request_num; ++i) {
        trt_multimodal::SharedVisGenHandle handle = runner->create_handle(
            gen_config,
            inputText
        );
        runner->enqueue_generate(handle);
        aio_handles.push_back(handle);
        std::this_thread::sleep_for(3s);
    }

    while (true) {
        bool all_gen_done = true;
        for (size_t i = 0; i < aio_handles.size(); ++i) {
            if (!aio_handles[i]->gen_finished.load()){
                all_gen_done = false;
            }
        }
        if (all_gen_done) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    auto overall_end = std::chrono::high_resolution_clock::now();

    runner->print_benchmark(
        overall_start, overall_end,
        aio_handles
    );

    std::cout << aio_handles[std::rand() % request_num]->generate_result.outputs_text[0] << std::endl;
    return 0;
}