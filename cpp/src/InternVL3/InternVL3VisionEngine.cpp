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
    tokens_per_patch = m_config.patch_tokens * m_config.embedding_dim; // [256, 896]
    max_out_elements = m_config.max_vis_batch * tokens_per_patch; // [max_vis_batch, 256, 896]
    pixels_per_patch = 3 * m_config.patch_size * m_config.patch_size; // [3, 448, 448]

    max_img_size = 512;
}

void InternVL3VisionEngine::initialize(size_t pool_size) {

    is_initialized.store(true);
    if (m_vis_session) m_vis_session.reset();
    if (d_images_pool) d_images_pool.reset();
    if (d_inputs_pool) d_inputs_pool.reset();
    if (d_outputs_pool) d_outputs_pool.reset();

    std::ifstream file(m_config.vis_engine_path, std::ios::binary);
    if (!file) throw std::runtime_error("Engine file not found: " +  m_config.vis_engine_path);

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    m_vis_session = std::make_unique<VisionSession>(
        buffer, pool_size, "input",
        nvinfer1::Dims4{
            m_config.max_vis_batch, 3, 
            m_config.patch_size, m_config.patch_size
        }
    );
    d_images_pool = std::make_unique<VisionSlotPool>(pool_size, max_img_size * max_img_size * 3);
    d_inputs_pool = std::make_unique<VisionSlotPool>(pool_size, m_config.max_vis_batch * pixels_per_patch * sizeof(__half)); // if input is fp16, then __half
    d_outputs_pool = std::make_unique<VisionSlotPool>(pool_size, max_out_elements * sizeof(__half)); // if output is fp16, then __half
}

AspectRatio InternVL3VisionEngine::find_closest_aspect_ratio(float aspect_ratio, int min_num, int max_num, int image_size) {
    float best_ratio_diff = std::numeric_limits<float>::infinity();
    AspectRatio best_ratio = {1, 1};

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

struct VisCallbackCtx {
    VisionSession* vis_session; int ctx_idx;
    VisionSlotPool* image_pool; int image_idx;
    VisionSlotPool* input_pool; int input_idx;
    VisionSlotPool* output_pool; int output_idx;
    SharedVisGenHandle handle;
};

void InternVL3VisionEngine::extract_visual_features(
    SharedVisGenHandle& handle,
    bool is_sync
) {

    assert(is_initialized.load());
    std::vector<cv::Mat>& images = handle->visual_features.images;
    GenerateConfig& gen_config = handle->gen_config;
    VisualFeatures& vis_feats = handle->visual_features;

    vis_feats.start_proc = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<cv::Rect>> all_patch_rects_on_src(images.size());
    for (size_t img_idx = 0; img_idx < images.size(); ++img_idx) {
        const cv::Mat image = images[img_idx];
        
        float aspect_ratio = (float)image.cols / image.rows;
        AspectRatio target_ratio = find_closest_aspect_ratio(
            aspect_ratio, gen_config.min_patch, gen_config.max_patch, m_config.patch_size);

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
    int image_slot_idx = d_images_pool->acquire_slot();
    int input_slot_idx = d_inputs_pool->acquire_slot();
    int output_slot_idx = d_outputs_pool->acquire_slot();
    assert(input_slot_idx != -1 && output_slot_idx != -1);

    void* d_image = d_images_pool->get_address(image_slot_idx);
    void* d_input = d_inputs_pool->get_address(input_slot_idx);
    void* d_output = d_outputs_pool->get_address(output_slot_idx);

    int ctx_idx = m_vis_session->acquire_context();
    assert(ctx_idx != -1);

    nvinfer1::IExecutionContext* ctx = m_vis_session->get_context(ctx_idx);
    ctx->setTensorAddress("input", d_input);
    ctx->setTensorAddress("output", d_output);

    // Greedy Approach
    size_t cur_patch_count = 0;
    size_t done_patch_count = 0;
    for (size_t img_idx = 0; img_idx < images.size(); ++img_idx) {
        cv::Mat img = images[img_idx];
        
        if (img.cols > max_img_size || img.rows > max_img_size) {
            float scale = std::min((float)max_img_size / img.cols, (float)max_img_size / img.rows);
            cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_AREA);
        }

        assert(img.isContinuous());
        cudaMemcpyAsync(d_image, img.data, img.total() * 3, cudaMemcpyHostToDevice, m_stream);

        // If use pinned memory, dont use cudafreeasync in the back, use unregister
        // cudaHostRegister(img.data, img.total() * 3, cudaHostRegisterMapped);
        // cudaHostGetDevicePointer(&d_src_img, img.data, 0);

        assert(all_patch_rects_on_src[img_idx].size() <= m_config.max_vis_batch);

        if (cur_patch_count + all_patch_rects_on_src[img_idx].size() > m_config.max_vis_batch) {
            ctx->enqueueV3(m_stream);

            size_t cur_out_elements = cur_patch_count * tokens_per_patch;
            void* d_cur_output_offset = (uint16_t*)d_final_output + (done_patch_count * tokens_per_patch);
            
            launch_fp16_to_bf16(d_output, d_cur_output_offset, cur_out_elements, m_stream);

            done_patch_count += cur_patch_count;
            cur_patch_count = 0;
        }

        __half *d_patch_start = (__half*)d_input + (cur_patch_count * pixels_per_patch);

        launch_vlm_preprocess(
            (uint8_t*)d_image,
            d_patch_start, 
            img.cols, img.rows, 
            m_config.patch_size,
            all_patch_rects_on_src[img_idx],
            m_stream
        );

        cur_patch_count += all_patch_rects_on_src[img_idx].size();      
    }

    if (cur_patch_count > 0) {
        ctx->enqueueV3(m_stream);

        size_t cur_out_elements = cur_patch_count * tokens_per_patch;
        void* d_cur_output_offset = (uint16_t*)d_final_output + (done_patch_count * tokens_per_patch);

        launch_fp16_to_bf16(d_output, d_cur_output_offset, cur_out_elements, m_stream);
        done_patch_count += cur_patch_count;
        cur_patch_count = 0;
    }

    // [Asynchronous Memory Management]
    // Capture a local copy of m_stream to avoid dependency on 'this' pointer.
    // This ensures that even if the VisionEngine is destroyed, the shared_ptr's 
    // custom deleter can safely release the GPU memory via cudaFreeAsync 
    // without accessing a dangling 'this' pointer.
    cudaStream_t stream_for_deleter = m_stream;
    vis_feats.embeddings_ptr = std::shared_ptr<void>(d_final_output, [stream_for_deleter](void* ptr) {
        if (ptr) cudaFreeAsync(ptr, stream_for_deleter);
    });
    vis_feats.dtype = DataType::BF16;

    auto cuda_callback_ctx = new VisCallbackCtx{
        m_vis_session.get(), ctx_idx,
        d_images_pool.get(), image_slot_idx,
        d_inputs_pool.get(), input_slot_idx,
        d_outputs_pool.get(), output_slot_idx,
        handle
    };

    // CUDA Callback
    cudaLaunchHostFunc(m_stream, [](void* userData) {
        auto c = static_cast<VisCallbackCtx*>(userData);
        c->vis_session->release_context(c->ctx_idx);
        c->image_pool->release_slot(c->image_idx);
        c->input_pool->release_slot(c->input_idx);
        c->output_pool->release_slot(c->output_idx);
        c->handle->vis_finished.store(true);
        c->handle->visual_features.end_proc = std::chrono::high_resolution_clock::now();
        delete c;
    }, cuda_callback_ctx);

    if (is_sync) cudaStreamSynchronize(m_stream);
}

void InternVL3VisionEngine::enqueue_extract_visual_features(
    SharedVisGenHandle& handle
) {
    extract_visual_features(handle, false);
}

} // namespace trt_multimodal