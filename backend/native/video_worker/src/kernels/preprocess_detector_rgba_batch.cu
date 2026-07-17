/* Batched RGBA pitch -> RetinaFace NCHW BGR planar preprocessing.
 *
 * Operates on an array of device RGBA surfaces (e.g. NvBufSurface RGBA pitches)
 * and writes a contiguous Bx3x640x640 float tensor.  Color order is BGR with
 * mean subtraction (104, 117, 123) to match the image_runtime preprocess path.
 */
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

__global__ void preprocess_detector_rgba_batch_kernel(
    const uint8_t* const* __restrict__ d_surface_ptrs,
    const int* __restrict__ d_pitches,
    const int* __restrict__ d_widths,
    const int* __restrict__ d_heights,
    int n,
    float* __restrict__ d_out)
{
    const int total_per_image = 3 * kTargetSize * kTargetSize;
    const int total = n * total_per_image;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int image_idx = idx / total_per_image;
    const int rem = idx % total_per_image;
    const int c = rem / (kTargetSize * kTargetSize);
    const int rem2 = rem % (kTargetSize * kTargetSize);
    const int y = rem2 / kTargetSize;
    const int x = rem2 % kTargetSize;

    const int h = d_heights[image_idx];
    const int w = d_widths[image_idx];
    const int pitch = d_pitches[image_idx];
    const uint8_t* src = d_surface_ptrs[image_idx];

    float src_x = (static_cast<float>(x) + 0.5f) * static_cast<float>(w) / kTargetSize - 0.5f;
    float src_y = (static_cast<float>(y) + 0.5f) * static_cast<float>(h) / kTargetSize - 0.5f;

    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));
    float dx = src_x - x0;
    float dy = src_y - y0;

    // Source RGBA: src_c selects source channel for output BGR channel c.
    int src_c = (c == 0) ? 2 : (c == 1 ? 1 : 0);

    auto fetch = [&](int yy, int xx) -> float {
        if (yy < 0 || yy >= h || xx < 0 || xx >= w) return 0.0f;
        const uint8_t* p = src + yy * pitch + xx * 4;
        return static_cast<float>(p[src_c]);
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

    d_out[idx] = val - kMean[c];
}

extern "C" cudaError_t mergenvision_preprocess_detector_rgba_batch(
    const uint8_t* const* d_surface_ptrs,
    const int* d_pitches,
    const int* d_widths,
    const int* d_heights,
    int n,
    float* d_out,
    cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    int total = n * 3 * kTargetSize * kTargetSize;
    constexpr int threads = 256;
    int blocks = (total + threads - 1) / threads;
    preprocess_detector_rgba_batch_kernel<<<blocks, threads, 0, stream>>>(
        d_surface_ptrs, d_pitches, d_widths, d_heights, n, d_out);
    return cudaGetLastError();
}

} // namespace mergenvision
