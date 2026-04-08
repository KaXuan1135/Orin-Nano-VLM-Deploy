#include <memory>
#include <cassert>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "TrtMultimodalRunner/InternVL3/vision_utils.h"
#include "TrtMultimodalRunner/InternVL3/InternVL3VisionEngine.hpp"

using namespace nvinfer1;

namespace trt_multimodal {

InternVL3VisionEngine::InternVL3VisionEngine(
    const ModelConfig& config,
    const cudaStream_t& stream
): m_config(config), m_stream(stream) {

    std::ifstream file(m_config.vis_engine_path, std::ios::binary);
    if (!file) throw std::runtime_error("Engine file not found: " +  m_config.vis_engine_path);

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    vis_engine = std::make_unique<VisionSession>();
    vis_engine->runtime = nvinfer1::createInferRuntime(gLogger);
    vis_engine->engine = vis_engine->runtime->deserializeCudaEngine(buffer.data(), buffer.size());
    vis_engine->context = vis_engine->engine->createExecutionContext();

    size_t tokens_per_patch = m_config.patch_token_size * m_config.embedding_dim; // [256, 896]
    size_t max_out_elements = m_config.max_vis_batch * tokens_per_patch; // [max_vis_batch, 256, 896]
    size_t pixels_per_patch = 3 * 448 * 448; // [3, 448, 448], previously 448 patch size get from gen_config, but necessary? 

    int max_vis_inflight = 2;
    int num_slots_for_input = 4;

    // output only need to follow max_vis_inflight, but for input there need to more for buffer (wait, i think no need?)
    d_inputs_pool.init_slots(max_vis_inflight, m_config.max_vis_batch * pixels_per_patch * sizeof(__half)); // if input is fp16, then __half
    d_outputs_pool.init_slots(max_vis_inflight, max_out_elements * sizeof(__half)); // if output is fp16, then __half

}

AspectRatio InternVL3VisionEngine::find_closest_aspect_ratio(float aspect_ratio, int min_num, int max_num, int image_size) {
    float best_ratio_diff = std::numeric_limits<float>::infinity();
    AspectRatio best_ratio = {1, 1};

    // 预计算所有可能的 target_ratios (1*1, 1*2, 2*1 等)
    for (int n = min_num; n <= max_num; ++n) {
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (i * j <= max_num && i * j >= min_num) {
                    float target_aspect_ratio = (float)i / j;
                    float ratio_diff = std::abs(aspect_ratio - target_aspect_ratio);
                    if (ratio_diff < best_ratio_diff) {
                        best_ratio_diff = ratio_diff;
                        best_ratio = {i, j};
                    } else if (ratio_diff == best_ratio_diff) {
                        // 如果比例相同，优先选择面积覆盖大的（对应 Python 代码逻辑）
                        if (i * j > best_ratio.width * best_ratio.height) {
                            best_ratio = {i, j};
                        }
                    }
                }
            }
        }
    }
    return best_ratio;
}

void InternVL3VisionEngine::extract_visual_features(
    const std::vector<cv::Mat>& images,
    const GenerateConfig& gen_config,
    VisualFeatures& vis_feats,
    const bool sync
) {
    vis_feats.start_proc = std::chrono::high_resolution_clock::now();
    vis_engine->context->setInputShape(
        "input", 
        nvinfer1::Dims4{
            m_config.max_vis_batch, 
            3, 
            gen_config.patch_size, 
            gen_config.patch_size
        }
    );

    std::vector<std::vector<cv::Rect>> all_patch_rects_on_src(images.size());
    for (size_t img_idx = 0; img_idx < images.size(); ++img_idx) {
        const cv::Mat image = images[img_idx];
        
        float aspect_ratio = (float)image.cols / image.rows;
        AspectRatio target_ratio = find_closest_aspect_ratio(
            aspect_ratio, gen_config.min_patch, gen_config.max_patch, gen_config.patch_size);

        float sw = (float)image.cols / target_ratio.width;
        float sh = (float)image.rows / target_ratio.height;
        
        int blocks = target_ratio.width * target_ratio.height;
        for (int i = 0; i < blocks; ++i) {
            int ix = i % target_ratio.width;
            int iy = i / target_ratio.width;
            all_patch_rects_on_src[img_idx].push_back(cv::Rect(ix * sw, iy * sh, sw, sh));
        }
        
        if (gen_config.use_thumbnail && blocks != 1) {
            all_patch_rects_on_src[img_idx].push_back(cv::Rect(0, 0, image.cols, image.rows));
        }
        
        vis_feats.image_patch_counts.push_back(all_patch_rects_on_src[img_idx].size());
    }
    
    /**
    Input / Output of Vision Engine : fp16
    Input of LLM Engine : bf16
    */
    void *d_final_output = nullptr;
    cudaMallocAsync(&d_final_output, vis_feats.total_patches() * tokens_per_patch * sizeof(uint16_t), m_stream); //bf16

    size_t current_patch_global_idx = 0;
    // int input_slot_idx = d_inputs_pool.acquire_slot();
    // int output_slot_idx = d_outputs_pool.acquire_slot();

    // void* d_input = d_inputs_pool.get_address(input_slot_idx);
    // void* d_output = d_outputs_pool.get_address(output_slot_idx);
    // assert(input_slot_idx != -1);
    // assert(output_slot_idx != -1);
    // vis_engine->context->setTensorAddress("input", d_input);
    // vis_engine->context->setTensorAddress("output", d_output);

    void *d_input = nullptr, *d_output = nullptr;
    cudaMallocAsync(&d_input, vis_feats.total_patches() * pixels_per_patch * sizeof(__half), m_stream);
    cudaMallocAsync(&d_output, max_out_elements * sizeof(uint16_t), m_stream);



    // // Greedy Approach
    // size_t cur_patch_count = 0;
    // size_t done_patch_count = 0;
    // for (size_t img_idx = 0; img_idx < images.size(); ++img_idx) {
    //     const cv::Mat& img = images[img_idx];
    //     assert(img.isContinuous());
        
    //     uint8_t* d_src_img = nullptr;
    //     size_t img_bytes = img.total() * 3;

    //     cudaMallocAsync(&d_src_img, img_bytes, m_stream);
    //     cudaMemcpyAsync(d_src_img, img.data, img_bytes, cudaMemcpyHostToDevice, m_stream);

    //     // If use pinned memory, dont use cudafreeasync in the back, use unregister
    //     // cudaHostRegister(img.data, img_bytes, cudaHostRegisterMapped);
    //     // cudaHostGetDevicePointer(&d_src_img, img.data, 0);

    //     assert(all_patch_rects_on_src[img_idx].size() <= m_config.max_vis_batch);

    //     if (cur_patch_count + all_patch_rects_on_src[img_idx].size() > m_config.max_vis_batch) {
    //         assert(false);

    //         size_t cur_out_elements = cur_patch_count * tokens_per_patch;
    //         void* d_cur_output_offset = (uint16_t*)d_final_output + (done_patch_count * tokens_per_patch);
            
    //         vis_engine->context->enqueueV3(m_stream);
    //         launch_fp16_to_bf16(d_output, d_cur_output_offset, cur_out_elements, m_stream);

    //         done_patch_count += cur_patch_count;
    //         cur_patch_count = 0;
    //     }

    //     __half *d_patch_start = (__half*)d_input + (cur_patch_count * pixels_per_patch);

    //     launch_vlm_preprocess(
    //         d_src_img, 
    //         d_patch_start, 
    //         img.cols, img.rows, 
    //         gen_config.patch_size,
    //         all_patch_rects_on_src[img_idx],
    //         m_stream
    //     );

    //     cur_patch_count += all_patch_rects_on_src[img_idx].size();
    //     cudaFreeAsync(d_src_img, m_stream);        
    // }

    // if (cur_patch_count > 0) {
        
    //     size_t cur_out_elements = cur_patch_count * tokens_per_patch;
    //     void* d_cur_output_offset = (uint16_t*)d_final_output + (done_patch_count * tokens_per_patch);
        
    //     vis_engine->context->enqueueV3(m_stream);
    //     launch_fp16_to_bf16(d_output, d_cur_output_offset, cur_out_elements, m_stream);
    //     done_patch_count += cur_patch_count;
    //     cur_patch_count = 0;
    // }

    for (size_t img_idx = 0; img_idx < images.size(); ++img_idx) {
        const cv::Mat& img = images[img_idx];
        assert(img.isContinuous());
        
        uint8_t* d_src_img = nullptr;
        size_t img_bytes = img.total() * 3;

        cudaMallocAsync(&d_src_img, img_bytes, m_stream);
        cudaMemcpyAsync(d_src_img, img.data, img_bytes, cudaMemcpyHostToDevice, m_stream);

        // If use pinned memory, dont use cudafreeasync in the back, use unregister
        // cudaHostRegister(img.data, img_bytes, cudaHostRegisterMapped);
        // cudaHostGetDevicePointer(&d_src_img, img.data, 0);

        __half* d_patch_start = (__half*)d_input + (current_patch_global_idx * pixels_per_patch);
        
        launch_vlm_preprocess(
            d_src_img, 
            d_patch_start, 
            img.cols, img.rows, 
            gen_config.patch_size,
            all_patch_rects_on_src[img_idx],
            m_stream
        );

        current_patch_global_idx += all_patch_rects_on_src[img_idx].size();
        cudaFreeAsync(d_src_img, m_stream);
    }

    for (size_t i = 0; i < vis_feats.total_patches(); i += m_config.max_vis_batch) {

        size_t cur_batch_size = std::min(static_cast<size_t>(m_config.max_vis_batch), vis_feats.total_patches() - i);
        size_t cur_out_elements = cur_batch_size * tokens_per_patch;

        void* d_current_output_offset = (uint16_t*)d_final_output + (i * tokens_per_patch);

        vis_engine->context->setTensorAddress("input", (uint16_t*)d_input + i * pixels_per_patch);
        vis_engine->context->setTensorAddress("output", d_output);
        vis_engine->context->enqueueV3(m_stream);
        std::cout << "hi" << std::endl;

        launch_fp16_to_bf16(d_output, d_current_output_offset, cur_out_elements, m_stream);
    }





    // [Asynchronous Memory Management]
    // We capture a local copy of m_stream to avoid dependency on 'this' pointer.
    // This ensures that even if the VisionEngine is destroyed, the shared_ptr's 
    // custom deleter can safely release the GPU memory via cudaFreeAsync 
    // without accessing a dangling 'this' pointer.
    cudaStream_t stream_for_deleter = m_stream;
    vis_feats.embeddings_ptr = std::shared_ptr<void>(d_final_output, [stream_for_deleter](void* ptr) {
        if (ptr) cudaFreeAsync(ptr, stream_for_deleter);
    });
    vis_feats.dtype = DataType::BF16;

    if (sync) cudaStreamSynchronize(m_stream);
    vis_feats.end_proc = std::chrono::high_resolution_clock::now();
}

void InternVL3VisionEngine::enqueue_extract_visual_features(
    const std::vector<cv::Mat>& images,
    const GenerateConfig& gen_config,
    SharedVisGenHandle& handle
) {
    extract_visual_features(
        images,
        gen_config,
        handle->visual_features,
        true
    );
    handle->vis_finished.store(true);
}

} // namespace trt_multimodal