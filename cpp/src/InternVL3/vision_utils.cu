#include "TrtMultimodalRunner/InternVL3/vision_utils.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__global__ void fp16_to_bf16_kernel(const __half* input, __nv_bfloat16* output, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float f = __half2float(input[i]);
        output[i] = __float2bfloat16(f);
    }
}

extern "C" void launch_fp16_to_bf16(const void* input, void* output, size_t n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fp16_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
        (const __half*)input, 
        (__nv_bfloat16*)output, 
        n
    );
}

__global__ void fp32_to_fp16_kernel(const float* input, __half* output, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = __float2half(input[i]);
    }
}

extern "C" void launch_fp32_to_fp16(const void* input, void* output, size_t n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    fp32_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
        (const float*)input, 
        (__half*)output, 
        n
    );
}