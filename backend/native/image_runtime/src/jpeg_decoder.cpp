#include "jpeg_decoder.h"
#include "util.h"
#include <cstring>

namespace mergenvision {

JpegDecoder::JpegDecoder() = default;

JpegDecoder::~JpegDecoder() {
    if (state_) {
        nvjpegJpegStateDestroy(state_);
    }
    if (handle_) {
        nvjpegDestroy(handle_);
    }
}

bool JpegDecoder::init(std::string* error) {
    nvjpegStatus_t status = nvjpegCreateSimple(&handle_);
    if (status != NVJPEG_STATUS_SUCCESS) {
        if (error) *error = "nvjpegCreateSimple failed";
        return false;
    }
    status = nvjpegJpegStateCreate(handle_, &state_);
    if (status != NVJPEG_STATUS_SUCCESS) {
        if (error) *error = "nvjpegJpegStateCreate failed";
        return false;
    }
    initialized_ = true;
    return true;
}

bool JpegDecoder::decode(const void* data,
                         std::size_t length,
                         int* out_width,
                         int* out_height,
                         unsigned char** d_rgb,
                         cudaStream_t stream,
                         std::string* error) {
    if (!initialized_) {
        if (error) *error = "JpegDecoder not initialized";
        return false;
    }
    if (!data || length == 0) {
        if (error) *error = "empty JPEG buffer";
        return false;
    }

    int n_components = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT] = {0};
    int heights[NVJPEG_MAX_COMPONENT] = {0};

    nvjpegStatus_t status = nvjpegGetImageInfo(
        handle_,
        static_cast<const unsigned char*>(data),
        length,
        &n_components,
        &subsampling,
        widths,
        heights);
    if (status != NVJPEG_STATUS_SUCCESS) {
        if (error) *error = "nvjpegGetImageInfo failed; buffer may not be a valid JPEG";
        return false;
    }
    if (widths[0] <= 0 || heights[0] <= 0) {
        if (error) *error = "invalid JPEG dimensions";
        return false;
    }

    int w = widths[0];
    int h = heights[0];
    size_t rgb_bytes = static_cast<size_t>(w) * h * 3;

    unsigned char* d_buf = nullptr;
    CU_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buf), rgb_bytes));

    nvjpegImage_t img{};
    img.channel[0] = d_buf;
    img.pitch[0] = w * 3;

    status = nvjpegDecode(
        handle_,
        state_,
        static_cast<const unsigned char*>(data),
        length,
        NVJPEG_OUTPUT_RGBI,
        &img,
        stream);
    if (status != NVJPEG_STATUS_SUCCESS) {
        CU_CHECK(cudaFree(d_buf));
        if (error) *error = "nvjpegDecode failed";
        return false;
    }

    *out_width = w;
    *out_height = h;
    *d_rgb = d_buf;
    return true;
}

} // namespace mergenvision
