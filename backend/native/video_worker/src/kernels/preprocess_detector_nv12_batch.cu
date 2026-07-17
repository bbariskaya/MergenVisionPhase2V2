/* Fused NV12/NV21 pitch -> RetinaFace NCHW BGR planar preprocessing.
 *
 * Reads per-frame Y and interleaved UV planes from NvBufSurface-style device
 * memory, bilinearly resamples to 640x640, converts YUV to RGB/BGR, applies
 * the same channel-mean subtraction (104,117,123) and writes a contiguous
 * Bx3x640x640 float tensor.  This matches the image_runtime RetinaFace
 * preprocess contract used by the shipped engines.
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <math.h>

namespace mergenvision {

constexpr int kTargetSize = 640;
__constant__ float kMean[3] = {104.0f, 117.0f, 123.0f};

struct YuvToRgbMatrix {
    float y;      // coefficient for raw Y
    float cb;     // coefficient for raw Cb
    float cr;     // coefficient for raw Cr
    float bias;   // constant offset in 0..255 space
};

// Modes: 0=BT.601 limited NV12, 1=BT.709 limited NV12,
//        2=BT.601 full NV12,    3=BT.709 full NV12,
//        4=BT.601 limited NV21, 5=BT.709 limited NV21,
//        6=BT.601 full NV21,    7=BT.709 full NV21.
// For NV21 the Cb/Cr samples are swapped at fetch time.
__constant__ YuvToRgbMatrix kMatrixR[8] = {
    {255.0f / 219.0f, 0.0f, 255.0f * 1.40200000f / 224.0f, -16.0f * 255.0f / 219.0f - 128.0f * 255.0f * 1.40200000f / 224.0f},                         // 0 BT.601 limited
    {255.0f / 219.0f, 0.0f, 255.0f * 1.57480000f / 224.0f, -16.0f * 255.0f / 219.0f - 128.0f * 255.0f * 1.57480000f / 224.0f},                         // 1 BT.709 limited
    {1.0f, 0.0f, 1.40200000f, -128.0f * 1.40200000f},                                                                                                  // 2 BT.601 full
    {1.0f, 0.0f, 1.57480000f, -128.0f * 1.57480000f},                                                                                                  // 3 BT.709 full
    {255.0f / 219.0f, 0.0f, 255.0f * 1.40200000f / 224.0f, -16.0f * 255.0f / 219.0f - 128.0f * 255.0f * 1.40200000f / 224.0f},                         // 4 BT.601 limited (NV21)
    {255.0f / 219.0f, 0.0f, 255.0f * 1.57480000f / 224.0f, -16.0f * 255.0f / 219.0f - 128.0f * 255.0f * 1.57480000f / 224.0f},                         // 5 BT.709 limited (NV21)
    {1.0f, 0.0f, 1.40200000f, -128.0f * 1.40200000f},                                                                                                  // 6 BT.601 full (NV21)
    {1.0f, 0.0f, 1.57480000f, -128.0f * 1.57480000f},                                                                                                  // 7 BT.709 full (NV21)
};

__constant__ YuvToRgbMatrix kMatrixG[8] = {
    {255.0f / 219.0f, -255.0f * 0.34413600f / 224.0f, -255.0f * 0.71413600f / 224.0f, -16.0f * 255.0f / 219.0f + 128.0f * 255.0f * (0.34413600f + 0.71413600f) / 224.0f},
    {255.0f / 219.0f, -255.0f * 0.18733000f / 224.0f, -255.0f * 0.46813000f / 224.0f, -16.0f * 255.0f / 219.0f + 128.0f * 255.0f * (0.18733000f + 0.46813000f) / 224.0f},
    {1.0f, -0.34413600f, -0.71413600f, 128.0f * (0.34413600f + 0.71413600f)},
    {1.0f, -0.18733000f, -0.46813000f, 128.0f * (0.18733000f + 0.46813000f)},
    {255.0f / 219.0f, -255.0f * 0.34413600f / 224.0f, -255.0f * 0.71413600f / 224.0f, -16.0f * 255.0f / 219.0f + 128.0f * 255.0f * (0.34413600f + 0.71413600f) / 224.0f},
    {255.0f / 219.0f, -255.0f * 0.18733000f / 224.0f, -255.0f * 0.46813000f / 224.0f, -16.0f * 255.0f / 219.0f + 128.0f * 255.0f * (0.18733000f + 0.46813000f) / 224.0f},
    {1.0f, -0.34413600f, -0.71413600f, 128.0f * (0.34413600f + 0.71413600f)},
    {1.0f, -0.18733000f, -0.46813000f, 128.0f * (0.18733000f + 0.46813000f)},
};

__constant__ YuvToRgbMatrix kMatrixB[8] = {
    {255.0f / 219.0f, 255.0f * 1.77200000f / 224.0f, 0.0f, -16.0f * 255.0f / 219.0f - 128.0f * 255.0f * 1.77200000f / 224.0f},
    {255.0f / 219.0f, 255.0f * 1.85563000f / 224.0f, 0.0f, -16.0f * 255.0f / 219.0f - 128.0f * 255.0f * 1.85563000f / 224.0f},
    {1.0f, 1.77200000f, 0.0f, -128.0f * 1.77200000f},
    {1.0f, 1.85563000f, 0.0f, -128.0f * 1.85563000f},
    {255.0f / 219.0f, 255.0f * 1.77200000f / 224.0f, 0.0f, -16.0f * 255.0f / 219.0f - 128.0f * 255.0f * 1.77200000f / 224.0f},
    {255.0f / 219.0f, 255.0f * 1.85563000f / 224.0f, 0.0f, -16.0f * 255.0f / 219.0f - 128.0f * 255.0f * 1.85563000f / 224.0f},
    {1.0f, 1.77200000f, 0.0f, -128.0f * 1.77200000f},
    {1.0f, 1.85563000f, 0.0f, -128.0f * 1.85563000f},
};

__device__ inline float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__global__ void preprocess_detector_nv12_batch_kernel(
    const uint8_t* const* __restrict__ d_y_ptrs,
    const uint8_t* const* __restrict__ d_uv_ptrs,
    const int* __restrict__ d_y_pitches,
    const int* __restrict__ d_uv_pitches,
    const int* __restrict__ d_widths,
    const int* __restrict__ d_heights,
    int n,
    int color_mode,
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
    const int y_pitch = d_y_pitches[image_idx];
    const int uv_pitch = d_uv_pitches[image_idx];
    const uint8_t* y_ptr = d_y_ptrs[image_idx];
    const uint8_t* uv_ptr = d_uv_ptrs[image_idx];

    float src_x = (static_cast<float>(x) + 0.5f) * static_cast<float>(w) / kTargetSize - 0.5f;
    float src_y = (static_cast<float>(y) + 0.5f) * static_cast<float>(h) / kTargetSize - 0.5f;

    // Luma sample coordinate.
    int y_x0 = static_cast<int>(floorf(src_x));
    int y_y0 = static_cast<int>(floorf(src_y));
    float y_dx = src_x - y_x0;
    float y_dy = src_y - y_y0;

    auto fetch_y = [&](int yy, int xx) -> float {
        if (yy < 0 || yy >= h || xx < 0 || xx >= w) return 0.0f;
        return static_cast<float>(y_ptr[yy * y_pitch + xx]);
    };

    float y00 = fetch_y(y_y0, y_x0);
    float y01 = fetch_y(y_y0, y_x0 + 1);
    float y10 = fetch_y(y_y0 + 1, y_x0);
    float y11 = fetch_y(y_y0 + 1, y_x0 + 1);
    float Y =
        (1.0f - y_dx) * (1.0f - y_dy) * y00 +
        y_dx * (1.0f - y_dy) * y01 +
        (1.0f - y_dx) * y_dy * y10 +
        y_dx * y_dy * y11;

    // Chroma sample coordinate (2x subsampled).
    int w2 = w / 2;
    int h2 = h / 2;
    float uv_x = src_x * 0.5f;
    float uv_y = src_y * 0.5f;
    int uv_x0 = static_cast<int>(floorf(uv_x));
    int uv_y0 = static_cast<int>(floorf(uv_y));
    float uv_dx = uv_x - uv_x0;
    float uv_dy = uv_y - uv_y0;

    auto fetch_uv_pair = [&](int yy, int xx, float& cb, float& cr) {
        if (yy < 0 || yy >= h2 || xx < 0 || xx >= w2) {
            cb = 128.0f;
            cr = 128.0f;
            return;
        }
        const uint8_t* p = uv_ptr + yy * uv_pitch + xx * 2;
        cb = static_cast<float>(p[0]);
        cr = static_cast<float>(p[1]);
    };

    float cb00, cr00, cb01, cr01, cb10, cr10, cb11, cr11;
    fetch_uv_pair(uv_y0, uv_x0, cb00, cr00);
    fetch_uv_pair(uv_y0, uv_x0 + 1, cb01, cr01);
    fetch_uv_pair(uv_y0 + 1, uv_x0, cb10, cr10);
    fetch_uv_pair(uv_y0 + 1, uv_x0 + 1, cb11, cr11);

    float Cb =
        (1.0f - uv_dx) * (1.0f - uv_dy) * cb00 +
        uv_dx * (1.0f - uv_dy) * cb01 +
        (1.0f - uv_dx) * uv_dy * cb10 +
        uv_dx * uv_dy * cb11;
    float Cr =
        (1.0f - uv_dx) * (1.0f - uv_dy) * cr00 +
        uv_dx * (1.0f - uv_dy) * cr01 +
        (1.0f - uv_dx) * uv_dy * cr10 +
        uv_dx * uv_dy * cr11;

    // NV21 semantics: swap chroma channels.
    if (color_mode >= 4) {
        float tmp = Cb;
        Cb = Cr;
        Cr = tmp;
    }

    const YuvToRgbMatrix& mR = kMatrixR[color_mode];
    const YuvToRgbMatrix& mG = kMatrixG[color_mode];
    const YuvToRgbMatrix& mB = kMatrixB[color_mode];

    float R = clampf(mR.y * Y + mR.cb * Cb + mR.cr * Cr + mR.bias, 0.0f, 255.0f);
    float G = clampf(mG.y * Y + mG.cb * Cb + mG.cr * Cr + mG.bias, 0.0f, 255.0f);
    float B = clampf(mB.y * Y + mB.cb * Cb + mB.cr * Cr + mB.bias, 0.0f, 255.0f);

    // Output order is BGR.
    float v;
    if (c == 0) v = B;
    else if (c == 1) v = G;
    else v = R;

    d_out[idx] = v - kMean[c];
}

extern "C" cudaError_t mergenvision_preprocess_detector_nv12_batch(
    const uint8_t* const* d_y_ptrs,
    const uint8_t* const* d_uv_ptrs,
    const int* d_y_pitches,
    const int* d_uv_pitches,
    const int* d_widths,
    const int* d_heights,
    int n,
    int color_mode,
    float* d_out,
    cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    int total = n * 3 * kTargetSize * kTargetSize;
    constexpr int threads = 256;
    int blocks = (total + threads - 1) / threads;
    preprocess_detector_nv12_batch_kernel<<<blocks, threads, 0, stream>>>(
        d_y_ptrs, d_uv_ptrs, d_y_pitches, d_uv_pitches,
        d_widths, d_heights, n, color_mode, d_out);
    return cudaGetLastError();
}

} // namespace mergenvision
