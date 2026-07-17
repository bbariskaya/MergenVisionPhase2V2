#pragma once

#include <cstddef>
#include <string>
#include <nvjpeg.h>
#include <cuda_runtime.h>

namespace mergenvision {

class JpegDecoder {
public:
    JpegDecoder();
    ~JpegDecoder();

    bool init(std::string* error);
    bool decode(const void* data,
                std::size_t length,
                int* out_width,
                int* out_height,
                unsigned char** d_rgb,
                cudaStream_t stream,
                std::string* error);

private:
    nvjpegHandle_t handle_ = nullptr;
    nvjpegJpegState_t state_ = nullptr;
    bool initialized_ = false;
};

} // namespace mergenvision
