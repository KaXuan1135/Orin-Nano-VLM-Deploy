#include <vector>
#include <string>
#include <thread>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "TrtMultimodalRunner/Types.hpp"
#include "TrtMultimodalRunner/IAsyncMultimodalRunner.hpp"

int main(int argc, char** argv) {

    int request_num = 100;

    std::string inputText = "Can you describe the 6 images?";

    std::vector<std::string> imagePaths = {
        "/home/pi/kx/sample_images/dog.jpg",
        "/home/pi/kx/sample_images/cat.jpg",
        "/home/pi/kx/sample_images/tiger.jpg",
        "/home/pi/kx/sample_images/wolf.jpg",
        "/home/pi/kx/sample_images/apple.jpg",
        "/home/pi/kx/sample_images/orange.jpg"
    };

    trt_multimodal::ModelConfig m_config = trt_multimodal::ModelConfig(
        trt_multimodal::ModelType::Type::INTERNVL3,
        "/mnt/sdcard/models/InternVL3-1B_i8"
    );

    trt_multimodal::GenerateConfig gen_config = trt_multimodal::GenerateConfig();
    gen_config.system_prompt = "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型, 英文名叫InternVL, 是一个有用无害的人工智能助手。";
    gen_config.image_prefix = "Image-$N$: ";
    gen_config.image_postfix = "\n";
    gen_config.max_new_tokens = 500;
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

    std::unique_ptr<trt_multimodal::IAsyncMultimodalRunner> runner = trt_multimodal::IAsyncMultimodalRunner::create(m_config);

    std::vector<cv::Mat> frames;
    for (auto path : imagePaths){
        cv::Mat rgbImg;
        cv::cvtColor(cv::imread(path), rgbImg, cv::COLOR_BGR2RGB);
        frames.push_back(std::move(rgbImg));
    }

    auto overall_start = std::chrono::high_resolution_clock::now();

    std::vector<trt_multimodal::SharedVisGenHandle> aio_handles;
    for (int i = 0; i < request_num; ++i) {
        aio_handles.push_back(runner->enqueue_generate(frames, inputText, gen_config));
    }

    while (true) {
        bool all_gen_done = true;
        for (size_t i = 0; i < aio_handles.size(); ++i) {
            if (!aio_handles[i]->gen_finished.load()){
                all_gen_done = false;
            }
            // auto& res = aio_handles[i]->generate_result;
            // if (!res.done_output.load()) {
            //     all_gen_done = false;
            // }
        }
        if (all_gen_done) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    auto overall_end = std::chrono::high_resolution_clock::now();

    runner->print_benchmark(
        overall_start, overall_end,
        aio_handles
    );

    std::cout << aio_handles[0]->generate_result.outputs_text[0] << std::endl;

    return 0;

}
