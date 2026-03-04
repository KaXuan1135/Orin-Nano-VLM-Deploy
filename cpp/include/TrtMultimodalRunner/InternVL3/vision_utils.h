#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void launch_fp16_to_bf16(const void* input, void* output, size_t n, cudaStream_t stream);
void launch_fp32_to_fp16(const void* input, void* output, size_t n, cudaStream_t stream);

#ifdef __cplusplus
}
#endif