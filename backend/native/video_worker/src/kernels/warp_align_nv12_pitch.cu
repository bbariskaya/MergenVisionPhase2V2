/* GPU five-point face alignment from pitched NV12/NVMM surfaces.
 *
 * Operates on a variable number of faces (n) that may come from one or more
 * frame surfaces. Each face is described by:
 *   - an affine matrix that maps destination (112x112) pixels back to the
 *     source luma plane (dst -> src);
 *   - a surface index selecting which d_y_ptrs / d_uv_ptrs base pointers to sample;
 *   - the luma/UV pitches, and source width/height.
 *
 * Output is a contiguous NCHW float tensor [n, 3, 112, 112] normalized by
 * subtracting 127.5 and dividing by 127.5 (ArcFace convention).
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdint.h>

namespace mergenvision {

constexpr float kArcMean = 127.5f;
constexpr float kArcStd = 127.5f;
constexpr int kOutSize = 112;

__device__ inline float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

// Bilinear sample from a uint8 plane with border constant = 0.
__device__ inline float sample_plane_bilinear(
    const uint8_t* __restrict__ plane,
    int h,
    int w,
    int pitch,
    float y,
    float x)
{
    float fx = floorf(x);
    float fy = floorf(y);
    float dx = x - fx;
    float dy = y - fy;
    int x0 = static_cast<int>(fx);
    int y0 = static_cast<int>(fy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    uint8_t v00 = 0, v01 = 0, v10 = 0, v11 = 0;
    auto fetch = [&](int yy, int xx, uint8_t& v) {
        if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
            v = plane[yy * pitch + xx];
        }
    };
    fetch(y0, x0, v00);
    fetch(y0, x1, v01);
    fetch(y1, x0, v10);
    fetch(y1, x1, v11);

    float w00 = (1.0f - dx) * (1.0f - dy);
    float w01 = dx * (1.0f - dy);
    float w10 = (1.0f - dx) * dy;
    float w11 = dx * dy;
    return w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
}

__device__ inline void nv12_to_rgb(float y, float u, float v,
                                   float* r, float* g, float* b)
{
    // BT.601 limited-range conversion.
    float yp = 1.164f * (y - 16.0f);
    float up = u - 128.0f;
    float vp = v - 128.0f;
    float rr = yp + 1.596f * vp;
    float gg = yp - 0.391f * up - 0.813f * vp;
    float bb = yp + 2.018f * up;
    *r = clampf(rr, 0.0f, 255.0f);
    *g = clampf(gg, 0.0f, 255.0f);
    *b = clampf(bb, 0.0f, 255.0f);
}

// One block per face: blockDim=(16,16,1), gridDim=(n,1,1).
__global__ void warp_align_nv12_pitch_kernel(
    const uint8_t* const* __restrict__ d_y_ptrs,
    const uint8_t* const* __restrict__ d_uv_ptrs,
    const int* __restrict__ d_surface_indices,
    const int* __restrict__ d_y_pitches,
    const int* __restrict__ d_uv_pitches,
    const int* __restrict__ d_widths,
    const int* __restrict__ d_heights,
    const float* __restrict__ d_matrices,
    int n,
    float* __restrict__ d_dst)
{
    int face = blockIdx.x;
    if (face >= n) return;

    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    const float* M = d_matrices + face * 6;
    float a = M[0];
    float b = M[3];
    float tx = M[2];
    float ty = M[5];
    float det = a * a + b * b;
    float inv_det = 0.0f;
    bool valid_m = det > 0.0f;
    if (valid_m) inv_det = 1.0f / det;

    int surf_idx = d_surface_indices[face];
    const uint8_t* y_base = d_y_ptrs[surf_idx];
    const uint8_t* uv_base = d_uv_ptrs[surf_idx];
    int y_pitch = d_y_pitches[face];
    int uv_pitch = d_uv_pitches[face];
    int w = d_widths[face];
    int h = d_heights[face];
    int uv_w = (w + 1) / 2;
    int uv_h = (h + 1) / 2;

    for (int y = tidy; y < kOutSize; y += blockDim.y) {
        for (int x = tidx; x < kOutSize; x += blockDim.x) {
            float sx = 0.0f, sy = 0.0f;
            if (valid_m) {
                sx = inv_det * (a * (x - tx) + b * (y - ty));
                sy = inv_det * (-b * (x - tx) + a * (y - ty));
            }
            float yy = sample_plane_bilinear(y_base, h, w, y_pitch, sy, sx);

            // UV plane is half-resolution and interleaved (U,V pairs).
            int uvx = static_cast<int>(roundf(sx * 0.5f));
            int uvy = static_cast<int>(roundf(sy * 0.5f));
            uvx = max(0, min(uvx, uv_w - 1));
            uvy = max(0, min(uvy, uv_h - 1));
            int uv_off = uvy * uv_pitch + uvx * 2;
            float uu = static_cast<float>(uv_base[uv_off]);
            float vv = static_cast<float>(uv_base[uv_off + 1]);

            float r, g, bval;
            nv12_to_rgb(yy, uu, vv, &r, &g, &bval);
            int base_idx = (face * 3 * kOutSize * kOutSize) + (y * kOutSize + x);
            d_dst[base_idx + 0 * kOutSize * kOutSize] = (r - kArcMean) / kArcStd;
            d_dst[base_idx + 1 * kOutSize * kOutSize] = (g - kArcMean) / kArcStd;
            d_dst[base_idx + 2 * kOutSize * kOutSize] = (bval - kArcMean) / kArcStd;
        }
    }
}

extern "C" cudaError_t mergenvision_warp_align_nv12_pitch(
    const uint8_t* const* d_y_ptrs,
    const uint8_t* const* d_uv_ptrs,
    const int* d_surface_indices,
    const int* d_y_pitches,
    const int* d_uv_pitches,
    const int* d_widths,
    const int* d_heights,
    const float* d_matrices,
    int n,
    float* d_dst,
    cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    dim3 block(16, 16, 1);
    dim3 grid(n, 1, 1);
    warp_align_nv12_pitch_kernel<<<grid, block, 0, stream>>>(
        d_y_ptrs, d_uv_ptrs, d_surface_indices,
        d_y_pitches, d_uv_pitches, d_widths, d_heights,
        d_matrices, n, d_dst);
    return cudaGetLastError();
}

} // namespace mergenvision
