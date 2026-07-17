#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <math.h>

namespace mergenvision {

constexpr int kTargetSize = 640;
__constant__ float kMean[3] = {104.0f, 117.0f, 123.0f};

__device__ inline float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__global__ void preprocess_retinaface_kernel(
    const uint8_t* __restrict__ rgb,
    int h,
    int w,
    float* __restrict__ out)
{
    int total = 3 * kTargetSize * kTargetSize;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int c = idx / (kTargetSize * kTargetSize);
    int rem = idx % (kTargetSize * kTargetSize);
    int y = rem / kTargetSize;
    int x = rem % kTargetSize;

    // Map output pixel center to source image coordinate.
    float src_x = (static_cast<float>(x) + 0.5f) * static_cast<float>(w) / kTargetSize - 0.5f;
    float src_y = (static_cast<float>(y) + 0.5f) * static_cast<float>(h) / kTargetSize - 0.5f;

    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));
    float dx = src_x - x0;
    float dy = src_y - y0;

    // Source channel index: output is BGR, source is RGB.
    int src_c = (c == 0) ? 2 : (c == 1 ? 1 : 0);

    auto fetch = [&](int yy, int xx) -> float {
        if (yy < 0 || yy >= h || xx < 0 || xx >= w) return 0.0f;
        return static_cast<float>(rgb[(yy * w + xx) * 3 + src_c]);
    };

    float v00 = fetch(y0, x0);
    float v01 = fetch(y0, x0 + 1);
    float v10 = fetch(y0 + 1, x0);
    float v11 = fetch(y0 + 1, x0 + 1);

    float val =
        (1.0f - dx) * (1.0f - dy) * v00 +
        dx * (1.0f - dy) * v01 +
        (1.0f - dx) * dy * v10 +
        dx * dy * v11;

    out[idx] = val - kMean[c];  // scale = 1.0
}

extern "C" int mergenvision_preprocess_retinaface(
    const uint8_t* d_rgb,
    int h,
    int w,
    float* d_out,
    cudaStream_t stream)
{
    if (h <= 0 || w <= 0) return 0;
    int total = 3 * kTargetSize * kTargetSize;
    constexpr int threads = 256;
    int blocks = (total + threads - 1) / threads;
    preprocess_retinaface_kernel<<<blocks, threads, 0, stream>>>(d_rgb, h, w, d_out);
    return cudaGetLastError();
}

} // namespace mergenvision
