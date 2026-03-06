#include <cuda_bf16.h>
#include <device_launch_parameters.h>

#include "TrtMultimodalRunner/InternVL3/vision_utils.h"

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

// 完成 Resize + Normalize + HWC2CHW + TypeCast
__global__ void vlm_preprocess_bilinear_kernel(
    const uint8_t* src, __half* dest,
    int src_w, int src_h, int patch_size,
    int crop_x, int crop_y, int crop_w, int crop_h,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < patch_size && dy < patch_size) {
        // 计算目标像素对应原图的浮点坐标
        float sx = (float)dx * crop_w / patch_size + crop_x;
        float sy = (float)dy * crop_h / patch_size + crop_y;

        // 找到周围的 4 个像素坐标
        int x0 = (int)floorf(sx);
        int y0 = (int)floorf(sy);
        int x1 = min(x0 + 1, src_w - 1);
        int y1 = min(y0 + 1, src_h - 1);

        // 计算插值权重
        float u = sx - x0;
        float v = sy - y0;

        // 读取 4 个点的 RGB 并进行线性插值
        auto get_pixel = [&](int x, int y, int c) {
            return (float)src[(y * src_w + x) * 3 + c];
        };

        float channels[3];
        for (int c = 0; c < 3; ++c) {
            float p00 = get_pixel(x0, y0, c);
            float p10 = get_pixel(x1, y0, c);
            float p01 = get_pixel(x0, y1, c);
            float p11 = get_pixel(x1, y1, c);

            // 双线性插值公式
            float val = (1 - u) * (1 - v) * p00 + u * (1 - v) * p10 +
                        (1 - u) * v * p01 + u * v * p11;
            
            // Normalize
            if (c == 0) channels[0] = (val / 255.0f - mean_r) / std_r;
            else if (c == 1) channels[1] = (val / 255.0f - mean_g) / std_g;
            else channels[2] = (val / 255.0f - mean_b) / std_b;
        }

        // HWC2CHW + TypeCast(fp32->fp16)
        int plane_size = patch_size * patch_size;
        int dest_idx = dy * patch_size + dx;
        dest[0 * plane_size + dest_idx] = __float2half(channels[0]);
        dest[1 * plane_size + dest_idx] = __float2half(channels[1]);
        dest[2 * plane_size + dest_idx] = __float2half(channels[2]);
    }
}

extern "C" void launch_vlm_preprocess(
    const uint8_t* d_src, __half* d_dest_base, 
    int src_w, int src_h, int patch_size,
    const std::vector<cv::Rect>& patches, // CPU 计算好的每个 patch 的位置
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((patch_size + block.x - 1) / block.x, (patch_size + block.y - 1) / block.y);

    for (int i = 0; i < patches.size(); ++i) {
        // 计算当前 patch 在大显存块中的起始地址
        __half* d_dest_patch = d_dest_base + (i * 3 * patch_size * patch_size);
        
        vlm_preprocess_bilinear_kernel<<<grid, block, 0, stream>>>(
            d_src, d_dest_patch,
            src_w, src_h,
            patch_size,
            patches[i].x, patches[i].y,
            patches[i].width, patches[i].height,
            0.485f, 0.456f, 0.406f,
            0.229f, 0.224f, 0.225f
        );
    }
}