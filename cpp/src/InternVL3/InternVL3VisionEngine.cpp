#include <memory>
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

    std::vector<float> transform_to_fp32_chw(const cv::Mat& patch) {
        cv::Mat float_img;
        patch.convertTo(float_img, CV_32FC3, 1.0 / 255.0); // ToTensor (0-1)

        // Normalize (ImageNet)
        cv::Scalar mean(0.485, 0.456, 0.406);
        cv::Scalar std(0.229, 0.224, 0.225);
        cv::subtract(float_img, mean, float_img);
        cv::divide(float_img, std, float_img);

        // HWC -> CHW (3, 448, 448)
        int area = patch.rows * patch.cols;
        std::vector<float> output(area * 3);
        std::vector<cv::Mat> channels(3);
        cv::split(float_img, channels);

        for (int i = 0; i < 3; ++i) {
            std::memcpy(output.data() + i * area, channels[i].data, area * sizeof(float));
        }
        return output;
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
                            // 逻辑：如果比例相同，优先选择面积覆盖大的（对应 Python 代码逻辑）
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

    std::vector<cv::Mat> InternVL3VisionEngine::dynamic_preprocess(const cv::Mat& image, int min_num, int max_num, int image_size, bool use_thumbnail) {
        int orig_width = image.cols;
        int orig_height = image.rows;
        float aspect_ratio = (float)orig_width / orig_height;

        AspectRatio target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, min_num, max_num, image_size);

        int target_width = image_size * target_aspect_ratio.width;
        int target_height = image_size * target_aspect_ratio.height;
        int blocks = target_aspect_ratio.width * target_aspect_ratio.height;

        cv::Mat resized_img;
        cv::resize(image, resized_img, cv::Size(target_width, target_height), 0, 0, cv::INTER_CUBIC);

        std::vector<cv::Mat> processed_images;
        for (int i = 0; i < blocks; ++i) {
            int x = (i % target_aspect_ratio.width) * image_size;
            int y = (i / target_aspect_ratio.width) * image_size;
            
            cv::Rect box(x, y, image_size, image_size);
            processed_images.push_back(resized_img(box).clone());
        }

        if (use_thumbnail && processed_images.size() != 1) {
            cv::Mat thumbnail;
            cv::resize(image, thumbnail, cv::Size(image_size, image_size), 0, 0, cv::INTER_CUBIC);
            processed_images.push_back(thumbnail);
        }
        return processed_images;
    }

    void InternVL3VisionEngine::extract_visual_features(
        const std::vector<cv::Mat>& images,
        const GenerateConfig& gen_config,
        VisualFeatures& vis_feats
    ) {

        std::vector<float> patches_overall; // [total_patches, 3, 448, 448]
        // CPU Preprocess TODO: preprocess on cuda
        for (const cv::Mat& image : images) {
            std::vector<cv::Mat> patches = dynamic_preprocess(
                image, 
                gen_config.min_patch,
                gen_config.max_patch,
                gen_config.patch_size,
                gen_config.use_thumbnail
            );

            for (const cv::Mat& patch : patches) {
                std::vector<float> flat_data = transform_to_fp32_chw(patch);
                patches_overall.insert(patches_overall.end(), flat_data.begin(), flat_data.end());
            }

            vis_feats.image_patch_counts.push_back(patches.size());
        }

        size_t tokens_per_patch = m_config.patch_token_size * m_config.embedding_dim; // [256, 896]
        size_t max_out_elements = m_config.max_vis_batch * tokens_per_patch; // [max_vis_batch, 256, 896]
        size_t pixels_per_patch = 3 * gen_config.patch_size * gen_config.patch_size; // [3, 448, 448]

        vis_engine->context->setInputShape(
            "input", 
            nvinfer1::Dims4{
                m_config.max_vis_batch, 
                3, 
                gen_config.patch_size, 
                gen_config.patch_size
            }
        );

        /**
        Preprocessed image in fp32, before inferening in Vision Engine need to be in fp16
        Output of Vision Engine is fp16, before inferencing in LLM Engine need to be in bf16
        */
        void *d_input_fp32 = nullptr, *d_input_fp16 = nullptr, *d_output_fp16 = nullptr, *d_all_outputs_bf16 = nullptr;
        cudaMallocAsync(&d_input_fp32, m_config.max_vis_batch * pixels_per_patch * sizeof(float), m_stream);
        cudaMallocAsync(&d_input_fp16, m_config.max_vis_batch * 3 * gen_config.patch_size * gen_config.patch_size * sizeof(__half), m_stream);
        cudaMallocAsync(&d_output_fp16, max_out_elements * sizeof(uint16_t), m_stream);
        cudaMallocAsync(&d_all_outputs_bf16, vis_feats.total_patches() * tokens_per_patch * sizeof(uint16_t), m_stream);

        for (size_t i = 0; i < vis_feats.total_patches(); i += m_config.max_vis_batch) {

            size_t cur_batch_size = std::min(static_cast<size_t>(m_config.max_vis_batch), vis_feats.total_patches() - i);
            size_t cur_out_elements = cur_batch_size * tokens_per_patch;

            void* d_current_output_offset = (uint16_t*)d_all_outputs_bf16 + (i * tokens_per_patch);

            float* current_patch_ptr = patches_overall.data() + (i * pixels_per_patch);
            size_t cur_in_elements = cur_batch_size * pixels_per_patch;

            cudaMemcpyAsync(d_input_fp32, current_patch_ptr, cur_in_elements * sizeof(float), cudaMemcpyHostToDevice, m_stream);
            launch_fp32_to_fp16(d_input_fp32, d_input_fp16, cur_in_elements, m_stream);

            vis_engine->context->setTensorAddress("input", d_input_fp16);
            vis_engine->context->setTensorAddress("output", d_output_fp16);
            vis_engine->context->enqueueV3(m_stream);

            launch_fp16_to_bf16(d_output_fp16, d_current_output_offset, cur_out_elements, m_stream);
        }

        cudaFreeAsync(d_input_fp32, m_stream);
        cudaFreeAsync(d_input_fp16, m_stream);
        cudaFreeAsync(d_output_fp16, m_stream);
        cudaStreamSynchronize(m_stream);

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

    }

    SharedVisHandle InternVL3VisionEngine::enqueue_extract_visual_features(
        const std::vector<cv::Mat>& images,
        const GenerateConfig& gen_config
    ) {

        auto handle = std::make_shared<VisHandle>();
        
        VisualFeatures vis_feats = VisualFeatures();
        std::vector<float> patches_overall;
        // CPU Preprocess TODO: preprocess on cuda
        for (const cv::Mat& image : images) {
            std::vector<cv::Mat> patches = dynamic_preprocess(
                image, 
                gen_config.min_patch,
                gen_config.max_patch,
                gen_config.patch_size,
                gen_config.use_thumbnail
            );

            for (const cv::Mat& patch : patches) {
                std::vector<float> flat_data = transform_to_fp32_chw(patch);
                patches_overall.insert(patches_overall.end(), flat_data.begin(), flat_data.end());
            }

            vis_feats.image_patch_counts.push_back(patches.size());
        }

        // 预先申请一整块连续显存，用于存放所有图片的特征
        size_t tokens_per_patch = m_config.patch_token_size * m_config.embedding_dim;
        void* d_all_outputs_bf16 = nullptr;
        cudaMallocAsync(&d_all_outputs_bf16, vis_feats.total_patches() * tokens_per_patch * sizeof(uint16_t), m_stream);

        vis_engine->context->setInputShape(
            "input", 
            nvinfer1::Dims4{
                m_config.max_vis_batch, 
                3, 
                gen_config.patch_size, 
                gen_config.patch_size
            }
        );

        // todo not neccessary 整除，might need to pad?
        for (size_t i = 0; i < vis_feats.total_patches(); i += m_config.max_vis_batch) {

            void* d_current_output_offset = (uint16_t*)d_all_outputs_bf16 + (i * tokens_per_patch);

            void *d_input = nullptr, *d_output_fp16 = nullptr, *d_output_bf16 = nullptr;
            size_t out_elements = m_config.max_vis_batch * m_config.patch_token_size * m_config.embedding_dim;

            cudaMallocAsync(&d_input, m_config.max_vis_batch * 3 * gen_config.patch_size * gen_config.patch_size * sizeof(__half), m_stream);
            cudaMallocAsync(&d_output_fp16, out_elements * sizeof(uint16_t), m_stream);
            cudaMallocAsync(&d_output_bf16, out_elements * sizeof(uint16_t), m_stream);

            std::vector<__half> host_input_half;
            // BUG: patches_overall need to 偏移
            // TODO : do precision convert in cuda, faster?
            for (float f : patches_overall) host_input_half.push_back(__float2half(f));
            cudaMemcpyAsync(d_input, host_input_half.data(), host_input_half.size() * sizeof(__half), cudaMemcpyHostToDevice);

            vis_engine->context->setTensorAddress("input", d_input);
            vis_engine->context->setTensorAddress("output", d_output_fp16);
            vis_engine->context->enqueueV3(m_stream);
            launch_fp16_to_bf16(d_output_fp16, d_current_output_offset, out_elements, m_stream);

            cudaFreeAsync(d_input, m_stream);
            cudaFreeAsync(d_output_fp16, m_stream);
        }

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

        handle->visual_features = vis_feats;
        handle->is_finished.store(true);
        return handle;

    }


} // namespace trt_multimodal