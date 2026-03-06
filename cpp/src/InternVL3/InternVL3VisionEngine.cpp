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

    int InternVL3VisionEngine::init(
        const ModelConfig& config,
        const cudaStream_t& stream
    ) {

        m_config = config;
        m_stream = stream;

        std::ifstream file(config.vis_engine_path, std::ios::binary);
        if (!file) throw std::runtime_error("Engine file not found: " +  config.vis_engine_path);

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);

        vis_engine = std::make_unique<VisionSession>();
        vis_engine->runtime = nvinfer1::createInferRuntime(gLogger);
        vis_engine->engine = vis_engine->runtime->deserializeCudaEngine(buffer.data(), buffer.size());
        vis_engine->context = vis_engine->engine->createExecutionContext();

        return 0;
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
        vis_engine->context->setInputShape(
            "input", 
            nvinfer1::Dims4{
                m_config.max_vis_batch, 
                3, 
                gen_config.patch_size, 
                gen_config.patch_size
            }
        );

        size_t tokens_per_patch = m_config.patch_token_size * m_config.embedding_dim; // [256, 896]
        size_t max_out_elements = m_config.max_vis_batch * tokens_per_patch; // [max_vis_batch, 256, 896]
        size_t pixels_per_patch = 3 * gen_config.patch_size * gen_config.patch_size; // [3, 448, 448]

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
        Before inferening in Vision Engine need to be in fp16
        Output of Vision Engine is fp16, before inferencing in LLM Engine need to be in bf16
        */
        void *d_input_fp16 = nullptr, *d_output_fp16 = nullptr, *d_all_outputs_bf16 = nullptr;
        cudaMallocAsync(&d_input_fp16, vis_feats.total_patches() * pixels_per_patch * sizeof(__half), m_stream);
        cudaMallocAsync(&d_output_fp16, max_out_elements * sizeof(uint16_t), m_stream);
        cudaMallocAsync(&d_all_outputs_bf16, vis_feats.total_patches() * tokens_per_patch * sizeof(uint16_t), m_stream);

        size_t current_patch_global_idx = 0;
        for (size_t img_idx = 0; img_idx < images.size(); ++img_idx) {
            const cv::Mat& img = images[img_idx];
            assert(img.isContinuous());
            
            uint8_t* d_src_img = nullptr;
            size_t img_bytes = img.total() * 3;

            cudaHostRegister(img.data, img_bytes, cudaHostRegisterMapped);
            cudaHostGetDevicePointer(&d_src_img, img.data, 0);

            __half* d_patch_start = (__half*)d_input_fp16 + (current_patch_global_idx * pixels_per_patch);
            
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

            void* d_current_output_offset = (uint16_t*)d_all_outputs_bf16 + (i * tokens_per_patch);

            vis_engine->context->setTensorAddress("input", (uint16_t*)d_input_fp16 + i * pixels_per_patch);
            vis_engine->context->setTensorAddress("output", d_output_fp16);
            vis_engine->context->enqueueV3(m_stream);

            launch_fp16_to_bf16(d_output_fp16, d_current_output_offset, cur_out_elements, m_stream);
        }

        cudaFreeAsync(d_input_fp16, m_stream);
        cudaFreeAsync(d_output_fp16, m_stream);
        
        // [Asynchronous Memory Management]
        // We capture a local copy of m_stream to avoid dependency on 'this' pointer.
        // This ensures that even if the VisionEngine is destroyed, the shared_ptr's 
        // custom deleter can safely release the GPU memory via cudaFreeAsync 
        // without accessing a dangling 'this' pointer.
        cudaStream_t stream_for_deleter = m_stream;
        vis_feats.embeddings_ptr = std::shared_ptr<void>(d_all_outputs_bf16, [stream_for_deleter](void* ptr) {
            if (ptr) {
                cudaFreeAsync(ptr, stream_for_deleter);
            }
        });
        vis_feats.dtype = DataType::BF16;

        if (sync) cudaStreamSynchronize(m_stream);
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
            false
        );
        handle->vis_finished.store(true);
    }

} // namespace trt_multimodal