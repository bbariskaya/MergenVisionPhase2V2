#include "glintr100_engine.h"
#include "model_profile.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace mergenvision {

class GlintR100Engine::Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[GlintR100Engine] " << msg << "\n";
        }
    }
};

namespace {

std::string dims_to_string(const nvinfer1::Dims& dims) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i) oss << ",";
        oss << dims.d[i];
    }
    oss << "]";
    return oss.str();
}

std::string dtype_to_string(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return "float32";
        case nvinfer1::DataType::kHALF: return "float16";
        case nvinfer1::DataType::kINT8: return "int8";
        case nvinfer1::DataType::kINT32: return "int32";
        case nvinfer1::DataType::kBOOL: return "bool";
        case nvinfer1::DataType::kUINT8: return "uint8";
        default: return "unknown";
    }
}

} // namespace

GlintR100Engine::GlintR100Engine() : logger_(std::make_unique<Logger>()) {}

GlintR100Engine::~GlintR100Engine() {
    delete context_;
    delete engine_;
    delete runtime_;
    if (d_input_buffer_) {
        cudaFree(d_input_buffer_);
    }
    if (d_output_buffer_) {
        cudaFree(d_output_buffer_);
    }
}

bool GlintR100Engine::load(int gpu_id, const std::string& engine_path,
                           std::string* error) {
    return load(gpu_id, ModelProfile(), engine_path, error);
}

bool GlintR100Engine::load(int gpu_id, const ModelProfile& profile,
                           const std::string& engine_path,
                           std::string* error) {
    if (loaded()) {
        if (error) *error = "engine already loaded";
        return false;
    }

    gpu_id_ = gpu_id;
    engine_path_ = engine_path;
    profile_ = profile;
    contract_ = Contract();  // start from hardcoded defaults
    if (!profile_.recognizer_input_name.empty()) {
        contract_.input_name = profile_.recognizer_input_name;
        contract_.output_name = profile_.recognizer_output_name;
        contract_.image_h = profile_.recognizer_input_h;
        contract_.image_w = profile_.recognizer_input_w;
        contract_.embedding_dim = profile_.recognizer_embedding_dim;
    }

    cudaError_t cuerr = cudaSetDevice(gpu_id_);
    if (cuerr != cudaSuccess) {
        if (error) *error = std::string("cudaSetDevice failed: ") + cudaGetErrorString(cuerr);
        return false;
    }

    std::ifstream file(engine_path_, std::ios::binary);
    if (!file) {
        if (error) *error = "failed to open engine file: " + engine_path_;
        return false;
    }
    std::vector<char> data((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
    if (data.empty()) {
        if (error) *error = "engine file is empty";
        return false;
    }

    runtime_ = nvinfer1::createInferRuntime(*logger_);
    if (!runtime_) {
        if (error) *error = "createInferRuntime failed";
        return false;
    }

    engine_ = runtime_->deserializeCudaEngine(data.data(), data.size());
    if (!engine_) {
        if (error) *error = "deserializeCudaEngine failed (TensorRT version mismatch?)";
        return false;
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        if (error) *error = "createExecutionContext failed";
        return false;
    }

    if (!validate_contract(error)) {
        return false;
    }

    if (!allocate_buffers(error)) {
        return false;
    }

    return true;
}

bool GlintR100Engine::validate_contract(std::string* error) {
    if (engine_->getNbIOTensors() != 2) {
        if (error) {
            *error = "expected 2 IO tensors, got " +
                     std::to_string(engine_->getNbIOTensors());
        }
        return false;
    }

    // Validate input tensor.
    bool found_input = false;
    bool found_output = false;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (contract_.input_name == name) {
            found_input = true;
            auto mode = engine_->getTensorIOMode(name);
            if (mode != nvinfer1::TensorIOMode::kINPUT) {
                if (error) *error = "input tensor is not marked INPUT";
                return false;
            }
            if (engine_->getTensorDataType(name) != contract_.input_dtype) {
                if (error) *error = "input dtype mismatch";
                return false;
            }
        } else if (contract_.output_name == name) {
            found_output = true;
            auto mode = engine_->getTensorIOMode(name);
            if (mode != nvinfer1::TensorIOMode::kOUTPUT) {
                if (error) *error = "output tensor is not marked OUTPUT";
                return false;
            }
            if (engine_->getTensorDataType(name) != contract_.output_dtype) {
                if (error) *error = "output dtype mismatch";
                return false;
            }
        }
    }

    if (!found_input) {
        if (error) *error = "input tensor '" + contract_.input_name + "' not found";
        return false;
    }
    if (!found_output) {
        if (error) *error = "output tensor '" + contract_.output_name + "' not found";
        return false;
    }

    if (engine_->getNbOptimizationProfiles() < 1) {
        if (error) *error = "engine has no optimization profile";
        return false;
    }

    nvinfer1::Dims input_max = engine_->getProfileShape(
        contract_.input_name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
    if (input_max.nbDims != contract_.input_rank) {
        if (error) {
            *error = "input rank mismatch: expected " +
                     std::to_string(contract_.input_rank) + " got " +
                     std::to_string(input_max.nbDims);
        }
        return false;
    }
    if (input_max.d[1] != 3 || input_max.d[2] != contract_.image_h ||
        input_max.d[3] != contract_.image_w) {
        if (error) {
            *error = "input shape mismatch: expected [*,3," +
                     std::to_string(contract_.image_h) + "," +
                     std::to_string(contract_.image_w) + "] got " +
                     dims_to_string(input_max);
        }
        return false;
    }

    // Output tensors do not expose profile shapes; use the static tensor shape.
    nvinfer1::Dims output_shape = engine_->getTensorShape(contract_.output_name.c_str());
    if (output_shape.nbDims != contract_.output_rank) {
        if (error) {
            *error = "output rank mismatch: expected " +
                     std::to_string(contract_.output_rank) + " got " +
                     std::to_string(output_shape.nbDims);
        }
        return false;
    }
    // Output dimension may be dynamic (-1); allow it as long as
    // declared embedding dimension is non-negative.
    if (output_shape.d[1] != -1 && output_shape.d[1] != contract_.embedding_dim) {
        if (error) {
            *error = "output dim mismatch: expected [*," +
                     std::to_string(contract_.embedding_dim) + "] got " +
                     dims_to_string(output_shape);
        }
        return false;
    }

    max_batch_ = input_max.d[0];
    if (max_batch_ < 1) {
        if (error) *error = "engine max batch < 1";
        return false;
    }

    return true;
}

bool GlintR100Engine::allocate_buffers(std::string* error) {
    if (max_batch_ < 1) {
        if (error) *error = "allocate_buffers called before max_batch set";
        return false;
    }

    constexpr int kInputElemsPerFace = 3 * 112 * 112;
    input_buffer_size_ = static_cast<size_t>(max_batch_) * kInputElemsPerFace;
    output_buffer_size_ = static_cast<size_t>(max_batch_) * contract_.embedding_dim;

    if (d_input_buffer_) { cudaFree(d_input_buffer_); d_input_buffer_ = nullptr; }
    if (d_output_buffer_) { cudaFree(d_output_buffer_); d_output_buffer_ = nullptr; }

    cudaError_t err = cudaMalloc(&d_input_buffer_,
                                 input_buffer_size_ * sizeof(float));
    if (err != cudaSuccess) {
        if (error) *error = std::string("cudaMalloc input buffer failed: ") +
                              cudaGetErrorString(err);
        return false;
    }
    err = cudaMalloc(&d_output_buffer_, output_buffer_size_ * sizeof(float));
    if (err != cudaSuccess) {
        if (error) {
            *error = std::string("cudaMalloc output buffer failed: ") +
                     cudaGetErrorString(err);
        }
        if (d_input_buffer_) { cudaFree(d_input_buffer_); d_input_buffer_ = nullptr; }
        return false;
    }

    return true;
}

bool GlintR100Engine::enqueue(int count, cudaStream_t stream, std::string* error) {
    if (!loaded()) {
        if (error) *error = "engine not loaded";
        return false;
    }
    if (count < 1 || count > max_batch_) {
        if (error) *error = "enqueue count " + std::to_string(count) +
                              " outside [1," + std::to_string(max_batch_) + "]";
        return false;
    }

    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = count;
    input_dims.d[1] = 3;
    input_dims.d[2] = 112;
    input_dims.d[3] = 112;

    if (!context_->setInputShape(contract_.input_name.c_str(), input_dims)) {
        if (error) *error = "setInputShape failed";
        return false;
    }

    if (!context_->setTensorAddress(contract_.input_name.c_str(),
                                    d_input_buffer_)) {
        if (error) *error = "setTensorAddress(input) failed";
        return false;
    }
    if (!context_->setTensorAddress(contract_.output_name.c_str(),
                                    d_output_buffer_)) {
        if (error) *error = "setTensorAddress(output) failed";
        return false;
    }

    if (!context_->enqueueV3(stream)) {
        if (error) *error = "enqueueV3 failed";
        return false;
    }

    return true;
}

} // namespace mergenvision
