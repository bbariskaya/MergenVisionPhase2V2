#include <cuda_runtime.h>
#include <cstdint>

__global__ void mv_convert_nchw_float_to_hwc_uint8_kernel(
    const float* __restrict__ d_src,
    uint8_t* __restrict__ d_dst,
    int n,
    int h,
    int w) {
    const int total = n * h * w;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int hw = h * w;
    const int batch = idx / hw;
    const int rem = idx % hw;
    const int y = rem / w;
    const int x = rem % w;

    const int dst_base = ((batch * h + y) * w + x) * 3;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const int src_idx = ((batch * 3 + c) * h + y) * w + x;
        float v = d_src[src_idx];
        if (v < 0.0f) {
            v = 0.0f;
        } else if (v > 255.0f) {
            v = 255.0f;
        }
        d_dst[dst_base + c] = static_cast<uint8_t>(v + 0.5f);
    }
}

extern "C" int mergenvision_convert_nchw_float_to_hwc_uint8(
    const float* d_src,
    uint8_t* d_dst,
    int n,
    int h,
    int w,
    cudaStream_t stream) {
    const int total = n * h * w;
    if (total <= 0) {
        return cudaSuccess;
    }

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    mv_convert_nchw_float_to_hwc_uint8_kernel<<<blocks, threads, 0, stream>>>(
        d_src, d_dst, n, h, w);
    return cudaGetLastError();
}
