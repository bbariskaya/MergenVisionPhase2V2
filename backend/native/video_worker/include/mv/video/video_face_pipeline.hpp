#pragma once

#include "mv/video/batch_assembler.hpp"
#include "mv/video/detection_mapper.hpp"
#include "mv/video/recognition_mapper.hpp"
#include "model_profile.h"

#include <cuda_runtime.h>
#include <gst/gst.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Forward declare image_runtime engine types.
namespace mergenvision {
class RetinaFaceEngine;
class RetinaFacePostproc;
class GlintR100Engine;
} // namespace mergenvision

namespace mergenvision::video {

struct PipelineStageMetrics {
    uint64_t preprocess_us = 0;
    uint64_t engine_enqueue_us = 0;
    uint64_t postproc_us = 0;
    uint64_t mapping_us = 0;
    uint64_t total_calls = 0;
};

// Batched video inference pipeline.  Owns the detector/recognizer TensorRT
// engines and the per-job CUDA stream.  All GPU work is asynchronous on the
// internal stream; callers must synchronize (via stream/event) before reading
// host-side results or releasing retained buffers.
class VideoFacePipeline {
public:
    VideoFacePipeline();
    ~VideoFacePipeline();

    // Non-copyable, non-movable.
    VideoFacePipeline(const VideoFacePipeline&) = delete;
    VideoFacePipeline& operator=(const VideoFacePipeline&) = delete;

    // Load TensorRT engines and allocate buffers. Returns false with diagnostic
    // message on failure.
    bool init(int gpu_id,
              const std::string& retinaface_engine_path,
              const std::string& glintr100_engine_path,
              std::string* error = nullptr);

    bool initialized() const { return initialized_; }
    int gpu_id() const { return gpu_id_; }
    cudaStream_t stream() const { return stream_; }

    // Run a batch of frames through RetinaFace.  The returned vector has the
    // same length and order as |batch.frames|.  No-face frames are represented
    // with an empty detection list so frame identity is never lost.
    std::vector<FrameDetections> infer_detector_batch(const InferenceFrameBatch& batch);

    // Run recognition on a deterministic ordered list of crops.  Currently a
    // placeholder: returns empty embeddings and marks recognition_eligible.
    // TODO(M5.2): implement real GlintR100 warp-align + L2 normalize.
    std::vector<EmbeddingResult> infer_recognition_batch(
        const std::vector<RecognitionCropRef>& crops);

    PipelineStageMetrics metrics() const { return metrics_; }

private:
    bool allocate_detector_buffers();

    int gpu_id_ = 0;
    cudaStream_t stream_ = nullptr;
    bool initialized_ = false;

    ModelProfile profile_;
    std::unique_ptr<mergenvision::RetinaFaceEngine> retina_engine_;
    std::unique_ptr<mergenvision::RetinaFacePostproc> retina_postproc_;
    std::unique_ptr<mergenvision::GlintR100Engine> glint_engine_;

    // GPU scratch space for detector preprocessing of one batch (max 8).
    float* d_detector_input_ = nullptr;
    size_t detector_input_bytes_ = 0;

    // Host/device arrays updated per batch for the RGBA preprocess kernel.
    std::vector<const uint8_t*> h_surface_ptrs_;
    std::vector<int> h_pitches_;
    std::vector<int> h_widths_;
    std::vector<int> h_heights_;
    uint8_t** d_surface_ptrs_ = nullptr;
    int* d_pitches_ = nullptr;
    int* d_widths_ = nullptr;
    int* d_heights_ = nullptr;

    // Host/device arrays for the NV12/NV21 fused preprocess kernel.
    std::vector<const uint8_t*> h_uv_ptrs_;
    std::vector<int> h_uv_pitches_;
    uint8_t** d_uv_ptrs_ = nullptr;
    int* d_uv_pitches_ = nullptr;

    PipelineStageMetrics metrics_{};
};

// Build a default model profile matching the shipped deepstream9 engine manifest.
ModelProfile make_default_model_profile();

} // namespace mergenvision::video
