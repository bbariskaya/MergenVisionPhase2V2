#include "webp_encoder.h"
#include <webp/encode.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mergenvision {

std::vector<uint8_t> encode_webp_rgb(
    const uint8_t* rgb,
    int width,
    int height,
    int stride,
    float quality) {
    if (!rgb || width <= 0 || height <= 0) {
        return {};
    }
    float q = std::max(1.0f, std::min(100.0f, quality));
    uint8_t* out = nullptr;
    size_t size = WebPEncodeRGB(rgb, width, height, stride, q, &out);
    if (size == 0 || !out) {
        return {};
    }
    std::vector<uint8_t> result(out, out + size);
    WebPFree(out);
    return result;
}

std::vector<uint8_t> encode_aligned_crop_webp(
    const float* nchw_rgb,
    int width,
    int height,
    float quality) {
    if (!nchw_rgb || width <= 0 || height <= 0) {
        return {};
    }
    std::vector<uint8_t> rgb(width * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int base = (y * width + x) * 3;
            int plane_offset = y * width + x;
            for (int c = 0; c < 3; ++c) {
                float v = nchw_rgb[c * width * height + plane_offset];
                int p = static_cast<int>(std::round(v * 127.5f + 127.5f));
                rgb[base + c] = static_cast<uint8_t>(std::max(0, std::min(255, p)));
            }
        }
    }
    return encode_webp_rgb(rgb.data(), width, height, width * 3, quality);
}

} // namespace mergenvision
