#pragma once
#include <vector>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#ifdef __cplusplus
extern "C" {
#endif

void launch_fp16_to_bf16(const void* input, void* output, size_t n, cudaStream_t stream);
void launch_fp32_to_fp16(const void* input, void* output, size_t n, cudaStream_t stream);
void launch_vlm_preprocess(
    const uint8_t* d_src, __half* d_dest_base, 
    int src_w, int src_h, int patch_size,
    const std::vector<cv::Rect>& patches,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif