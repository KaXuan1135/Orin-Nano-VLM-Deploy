#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include "TrtMultimodalRunner/IMultimodalRunner.hpp"
#include "TrtMultimodalRunner/InternVL3/InternVL3Runner.hpp"
#include "TrtMultimodalRunner/InternVL3/AsyncInternVL3Runner.hpp"

namespace py = pybind11;
using namespace trt_multimodal;

// --- 辅助函数 (放在模块宏外面即可) ---
cv::Mat numpy_to_mat(py::array_t<uint8_t> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 3) throw std::runtime_error("Image must be 3D (H,W,C)");
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);
    return mat.clone();
}

// std::vector<trt_multimodal::GenerateResult> py_batch_generate(
//     trt_multimodal::InternVL3Runner& runner,
//     const std::vector<std::vector<py::array_t<uint8_t>>>& py_images,
//     const std::vector<std::string>& user_prompt,
//     const std::vector<trt_multimodal::GenerateConfig>& gen_config
// ) {
//     std::vector<std::vector<cv::Mat>> images;
//     for (const auto& row : py_images) {
//         std::vector<cv::Mat> mat_row;
//         for (const auto& arr : row) mat_row.push_back(numpy_to_mat(arr));
//         images.push_back(mat_row);
//     }
    
//     std::vector<trt_multimodal::GenerateResult> results;
//     runner.batch_generate(images, user_prompt, gen_config, results);
//     return results;
// }

SharedVisGenHandle py_enqueue_chat(
    trt_multimodal::AsyncInternVL3Runner& runner,
    const std::vector<std::vector<py::array_t<uint8_t>>>& py_images,
    const std::vector<std::string>& user_prompt,
    const std::vector<trt_multimodal::GenerateConfig>& gen_config,
    const std::vector<SharedVisGenHandle>& prev_handles
) {
    std::vector<std::vector<cv::Mat>> images;
    for (const auto& row : py_images) {
        std::vector<cv::Mat> mat_row;
        for (const auto& arr : row) mat_row.push_back(numpy_to_mat(arr));
        images.push_back(mat_row);
    }

    return runner.enqueue_generate(images[0], user_prompt[0], gen_config[0], prev_handles);
}


// --- 绑定逻辑 (必须放在这里面！) ---
PYBIND11_MODULE(my_engine_binding, m) {
    m.doc() = "High-performance AI Engine Bindings";

    // 绑定 GenerateConfig
    py::class_<trt_multimodal::GenerateConfig>(m, "GenerateConfig")
        .def(py::init<>())
        .def_readwrite("max_new_tokens", &trt_multimodal::GenerateConfig::max_new_tokens)
        .def_readwrite("temperature", &trt_multimodal::GenerateConfig::temperature)
        .def_readwrite("streaming", &trt_multimodal::GenerateConfig::streaming);

    // 绑定 GenerateResult
    py::class_<trt_multimodal::GenerateResult>(m, "GenerateResult")
        .def_readwrite("output_text", &trt_multimodal::GenerateResult::outputs_text)
        .def_readwrite("token_ids", &trt_multimodal::GenerateResult::outputs_tokens);

    // 1. 绑定 ModelType 枚举 (如果它是 enum)
    py::enum_<ModelType::Type>(m, "ModelType")
        .value("INTERNVL3", ModelType::Type::INTERNVL3); // 对应你的实际枚举值

    // 2. 绑定 ModelConfig 结构体
    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<>()) // 暴露默认构造函数
        .def_readwrite("model_type", &ModelConfig::model_type)
        .def_readwrite("llm_engine_path", &ModelConfig::llm_engine_path)
        .def_readwrite("vis_engine_path", &ModelConfig::vis_engine_path)
        .def_readwrite("tokenizer_path", &ModelConfig::tokenizer_path)
        .def_readwrite("max_beam_width", &ModelConfig::max_beam_width)
        .def_readwrite("max_llm_batch", &ModelConfig::max_llm_batch)
        .def_readwrite("max_vis_batch", &ModelConfig::max_vis_batch)
        .def_readwrite("patch_token_size", &ModelConfig::patch_token_size)
        .def_readwrite("embedding_dim", &ModelConfig::embedding_dim);

    // 3. 绑定 Runner (这里直接接受 ModelConfig 对象)
    // py::class_<InternVL3Runner>(m, "InternVL3Runner")
    //     .def(py::init<const ModelConfig&>()) // 这一行直接匹配了 C++ 的构造函数
    //     .def("batch_generate", &py_batch_generate); 
 
    py::class_<AsyncInternVL3Runner>(m, "AsyncInternVL3Runner")
        .def(py::init<const ModelConfig&>())
        .def("enqueue_chat", &py_enqueue_chat);

    py::class_<VisGenHandle, SharedVisGenHandle>(m, "VisGenHandle")
        // 使用 getter 暴露状态，避免直接暴露 std::atomic
        .def_property_readonly("vis_finished", [](const VisGenHandle& h) { return h.vis_finished.load(); })
        .def_property_readonly("gen_finished", [](const VisGenHandle& h) { return h.gen_finished.load(); })
        
        // 关键：不要用 def_readwrite 暴露 last_outputs_text，必须用互斥锁保护的方法
        .def("pop_last_outputs_text", &VisGenHandle::pop_last_outputs_text);


}