#pragma once

#include "jpeg_decoder.h"
#include "model_profile.h"
#include "retinaface_engine.h"
#include "retinaface_postproc.h"
#include "glintr100_engine.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mergenvision {

struct FaceObservation {
    int detection_index = 0;
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    std::vector<float> landmarks5;  // 10 floats: x0,y0,...,x4,y4
    float detector_confidence = 0.0f;
    std::vector<float> embedding;   // 512 floats (L2-normalized)
    std::vector<uint8_t> aligned_crop_bytes;  // WebP
};

struct InferenceResult {
    int image_width = 0;
    int image_height = 0;
    std::vector<FaceObservation> detections;
};

class ExecutionSlot {
public:
    enum class State {
        uninitialized,
        initialized,
        unavailable,
        in_use,
    };

    ExecutionSlot(const ModelProfile& profile,
                  int device_id,
                  const std::string& retinaface_engine_path,
                  const std::string& glintr100_engine_path,
                  std::string* error);
    ~ExecutionSlot();

    State state() const { return state_; }
    bool available() const { return state_ == State::initialized; }
    void acquire() { state_ = State::in_use; }
    void release() { state_ = State::initialized; }

    bool infer_jpeg(const void* jpeg_data,
                    std::size_t jpeg_size,
                    InferenceResult* result,
                    std::string* error);

private:
    bool ensure_buffers(int max_faces, std::string* error);

    ModelProfile profile_;
    int device_id_ = 0;
    cudaStream_t stream_ = nullptr;

    JpegDecoder decoder_;
    std::unique_ptr<RetinaFaceEngine> retina_engine_;
    std::unique_ptr<RetinaFacePostproc> retina_postproc_;
    std::unique_ptr<GlintR100Engine> glint_engine_;

    // Reusable detector input.
    float* d_detector_input_ = nullptr;
    size_t detector_input_bytes_ = 0;

    // Reusable per-face buffers, sized to recognizer max batch.
    float* d_aligned_crops_ = nullptr;
    float* d_face_landmarks_ = nullptr;
    float* d_face_matrices_ = nullptr;
    int* d_kernel_status_ = nullptr;

    float* h_embeddings_ = nullptr;
    float* h_aligned_crops_ = nullptr;

    int max_faces_ = 0;
    State state_ = State::uninitialized;
};

} // namespace mergenvision
