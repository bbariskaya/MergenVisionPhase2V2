#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace mergenvision {

inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error at ") + file + ":" + std::to_string(line) +
            " -> " + cudaGetErrorString(err));
    }
}

#define CU_CHECK(x) mergenvision::cuda_check((x), __FILE__, __LINE__)

} // namespace mergenvision
