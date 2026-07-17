#pragma once

#include <cstdint>
#include <vector>

namespace mergenvision {

// Encode an RGB uint8 buffer (interleaved, width*height*3 bytes) as WebP.
// Returns empty vector on failure.
std::vector<uint8_t> encode_webp_rgb(
    const uint8_t* rgb,
    int width,
    int height,
    int stride,
    float quality);

// Convert a NCHW float buffer (RGB, normalized (x-127.5)/127.5) to interleaved
// uint8 RGB and encode as WebP.
std::vector<uint8_t> encode_aligned_crop_webp(
    const float* nchw_rgb,
    int width,
    int height,
    float quality);

} // namespace mergenvision
