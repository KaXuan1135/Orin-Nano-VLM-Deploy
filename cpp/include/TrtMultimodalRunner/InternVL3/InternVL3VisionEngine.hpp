#pragma once
#include <iostream>
#include <NvInfer.h>

#include "TrtMultimodalRunner/Types.hpp" 

namespace trt_multimodal {

class VisionSlotPool {

public:

    VisionSlotPool(size_t num_slots, size_t slot_size) {
        for (size_t i = 0; i < num_slots; ++i) {
            void *ptr = nullptr;
            cudaMalloc(&ptr, slot_size);
            m_slots.push_back({ptr, false});
            m_free_indices.push(i);
        }
    }

    ~VisionSlotPool() {
        for (auto& slot : m_slots) {
            if (slot.ptr) cudaFree(slot.ptr);
        }
    }

    int acquire_slot() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_free_indices.size() < 1) return -1;

        int idx = m_free_indices.front();
        m_free_indices.pop();
        m_slots[idx].occupied = true;

        return idx;
    }

    void release_slot(const int& idx) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_free_indices.push(idx);
        m_slots[idx].occupied = false;
    }

    void* get_address(int index) { 
        if (index == -1) return nullptr;
        return m_slots[index].ptr; 
    }

private:

    struct Slot{
        void* ptr;
        bool occupied;
    };

    std::vector<Slot> m_slots;
    std::queue<int> m_free_indices;
    std::mutex m_mutex;

};

class TRTLogger : public nvinfer1::ILogger {

private:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            std::string label;
            switch (severity) {
                case Severity::kINTERNAL_ERROR: label = "[FATAL]";   break;
                case Severity::kERROR:          label = "[ERROR]";   break;
                case Severity::kWARNING:        label = "[WARNING]"; break;
                case Severity::kINFO:           label = "[INFO]";    break;
                case Severity::kVERBOSE:        label = "[VERBOSE]"; break;
                default:                        label = "[UNKNOWN]"; break;
            }
            std::cout << "[TensorRT-VIS]" << label << " " << msg << std::endl;
        }
    }
};

class VisionSession {

public:

    VisionSession(
        const std::vector<char>& buffer,
        const size_t& num_ctx,
        const std::string& input_name,
        const nvinfer1::Dims4& input_dims
    ) {
        m_logger = TRTLogger();
        m_runtime = nvinfer1::createInferRuntime(m_logger);
        m_engine = m_runtime->deserializeCudaEngine(buffer.data(), buffer.size());;
        for (size_t i = 0; i < num_ctx; ++i) {
            m_contexts.push_back(m_engine->createExecutionContext());
            m_contexts[i]->setInputShape(input_name.c_str(), input_dims);
            m_free_context_indices.push(i);
        }
    }

    ~VisionSession() {
        for (auto* ctx : m_contexts) {
            if (ctx) delete ctx; // TensorRT 10.x 之后可以直接 delete
        }
        if (m_engine) delete m_engine;
        if (m_runtime) delete m_runtime;
    }

    int acquire_context() {
        std::lock_guard<std::mutex> lock(m_ctx_mutex);
        if (m_free_context_indices.empty()) return -1;
        int idx = m_free_context_indices.front();
        m_free_context_indices.pop();
        return idx;
    }

    void release_context(int idx) {
        std::lock_guard<std::mutex> lock(m_ctx_mutex);
        m_free_context_indices.push(idx);
    }

    nvinfer1::IExecutionContext* get_context(int idx) {
        return m_contexts[idx];
    }

private:
    TRTLogger m_logger;
    nvinfer1::IRuntime* m_runtime = nullptr;
    nvinfer1::ICudaEngine* m_engine = nullptr;
    
    std::vector<nvinfer1::IExecutionContext*> m_contexts;
    std::queue<int> m_free_context_indices;
    std::mutex m_ctx_mutex;
};

class InternVL3VisionEngine {
public:

    InternVL3VisionEngine(
        const ModelConfig& config,
        const cudaStream_t& stream
    );

    void initialize(size_t pool_size);

    void extract_visual_features(
        SharedVisGenHandle& handle,
        bool is_sync = true
    );

    void enqueue_extract_visual_features(
        SharedVisGenHandle& handle
    );

private:

    std::atomic<bool> is_initialized{false};
    std::unique_ptr<VisionSession> m_vis_session;
    ModelConfig m_config;
    cudaStream_t m_stream;

    size_t tokens_per_patch;
    size_t max_out_elements;
    size_t pixels_per_patch;
    size_t max_img_size;

    std::unique_ptr<VisionSlotPool> d_images_pool;
    std::unique_ptr<VisionSlotPool> d_inputs_pool;
    std::unique_ptr<VisionSlotPool> d_outputs_pool;

    AspectRatio find_closest_aspect_ratio(float aspect_ratio, int min_num, int max_num, int image_size);
};

} // namespace trt_multimodal