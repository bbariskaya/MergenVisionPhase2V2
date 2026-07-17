#include "mv/video/video_face_pipeline.hpp"

#include "mv/video/detection_mapper.hpp"
#include "retinaface_engine.h"
#include "retinaface_postproc.h"
#include "glintr100_engine.h"
#include "mergenvision_kernels.h"

#include <cuda_runtime.h>
#include <gst/gst.h>
#include <nvbufsurface.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>

namespace mergenvision::video {

extern "C" cudaError_t mergenvision_preprocess_detector_rgba_batch(
    const uint8_t* const* d_surface_ptrs,
    const int* d_pitches,
    const int* d_widths,
    const int* d_heights,
    int n,
    float* d_out,
    cudaStream_t stream);

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
    cudaStream_t stream);

namespace {

#define CU_CHECK(expr)                                                  \
    do {                                                                \
        cudaError_t err = (expr);                                       \
        if (err != cudaSuccess) {                                       \
            std::ostringstream oss;                                     \
            oss << "CUDA error: " << cudaGetErrorString(err)            \
                << " at " << __FILE__ << ":" << __LINE__;             \
            throw std::runtime_error(oss.str());                        \
        }                                                               \
    } while (0)

std::string color_format_name(int fmt) {
    switch (fmt) {
        case NVBUF_COLOR_FORMAT_GRAY8: return "GRAY8";
        case NVBUF_COLOR_FORMAT_YUV420: return "YUV420";
        case NVBUF_COLOR_FORMAT_YVU420: return "YVU420";
        case NVBUF_COLOR_FORMAT_NV12: return "NV12";
        case NVBUF_COLOR_FORMAT_NV12_ER: return "NV12_ER";
        case NVBUF_COLOR_FORMAT_NV21: return "NV21";
        case NVBUF_COLOR_FORMAT_NV21_ER: return "NV21_ER";
        case NVBUF_COLOR_FORMAT_YUV444: return "YUV444";
        case NVBUF_COLOR_FORMAT_RGBA: return "RGBA";
        case NVBUF_COLOR_FORMAT_BGRA: return "BGRA";
        case NVBUF_COLOR_FORMAT_ARGB: return "ARGB";
        case NVBUF_COLOR_FORMAT_ABGR: return "ABGR";
        case NVBUF_COLOR_FORMAT_RGB: return "RGB";
        case NVBUF_COLOR_FORMAT_BGR: return "BGR";
        case NVBUF_COLOR_FORMAT_NV12_10LE: return "NV12_10LE";
        case NVBUF_COLOR_FORMAT_NV12_12LE: return "NV12_12LE";
        case NVBUF_COLOR_FORMAT_YUV420_709: return "YUV420_709";
        case NVBUF_COLOR_FORMAT_NV12_709: return "NV12_709";
        case NVBUF_COLOR_FORMAT_NV12_709_ER: return "NV12_709_ER";
        case NVBUF_COLOR_FORMAT_YUV420_2020: return "YUV420_2020";
        case NVBUF_COLOR_FORMAT_NV12_2020: return "NV12_2020";
        case NVBUF_COLOR_FORMAT_RGBA_10_10_10_2_709: return "RGBA_10_10_10_2_709";
        case NVBUF_COLOR_FORMAT_BGRA_10_10_10_2_709: return "BGRA_10_10_10_2_709";
        default: return "UNKNOWN(" + std::to_string(fmt) + ")";
    }
}

bool is_rgba_format(int fmt) {
    return fmt == NVBUF_COLOR_FORMAT_RGBA ||
           fmt == NVBUF_COLOR_FORMAT_BGRA ||
           fmt == NVBUF_COLOR_FORMAT_ARGB ||
           fmt == NVBUF_COLOR_FORMAT_ABGR;
}

bool is_nv12_family(int fmt) {
    return fmt == NVBUF_COLOR_FORMAT_NV12 ||
           fmt == NVBUF_COLOR_FORMAT_NV12_ER ||
           fmt == NVBUF_COLOR_FORMAT_NV12_709 ||
           fmt == NVBUF_COLOR_FORMAT_NV12_709_ER ||
           fmt == NVBUF_COLOR_FORMAT_NV12_2020 ||
           fmt == NVBUF_COLOR_FORMAT_NV12_10LE ||
           fmt == NVBUF_COLOR_FORMAT_NV12_10LE_709 ||
           fmt == NVBUF_COLOR_FORMAT_NV12_10LE_2020 ||
           fmt == NVBUF_COLOR_FORMAT_NV21 ||
           fmt == NVBUF_COLOR_FORMAT_NV21_ER;
}

// Map NvBufSurfaceColorFormat to the fused kernel color mode.
// Modes: 0=BT.601 limited NV12, 1=BT.709 limited NV12,
//        2=BT.601 full NV12,    3=BT.709 full NV12,
//        4=BT.601 limited NV21, 5=BT.709 limited NV21,
//        6=BT.601 full NV21,    7=BT.709 full NV21.
int nv12_color_mode(int fmt) {
    switch (fmt) {
        case NVBUF_COLOR_FORMAT_NV21: return 4;
        case NVBUF_COLOR_FORMAT_NV21_ER: return 6;
        case NVBUF_COLOR_FORMAT_NV12: return 0;
        case NVBUF_COLOR_FORMAT_NV12_ER: return 2;
        case NVBUF_COLOR_FORMAT_NV12_709:
        case NVBUF_COLOR_FORMAT_NV12_2020:
        case NVBUF_COLOR_FORMAT_NV12_10LE_709:
        case NVBUF_COLOR_FORMAT_NV12_10LE_2020: return 1;
        case NVBUF_COLOR_FORMAT_NV12_709_ER: return 3;
        default: return -1;
    }
}

ModelProfile build_default_profile() {
    ModelProfile p;
    p.model_version = "retinaface_r50_glintr100_v1";
    p.preprocess_version = "deepstream9_rgba_pitch_v1";

    p.detector_input_name = "input";
    p.detector_loc_name = "loc";
    p.detector_conf_name = "conf";
    p.detector_landms_name = "landms";
    p.detector_input_size = 640;
    p.detector_conf_threshold = 0.5f;
    p.detector_nms_threshold = 0.4f;
    p.detector_max_candidates = 300;
    p.detector_anchor_strides = {8, 16, 32};
    p.detector_anchor_sizes = {{16, 32}, {64, 128}, {256, 512}};
    p.detector_anchor_ratios = {1.0f};
    p.detector_variances = {0.1f, 0.2f};

    p.recognizer_input_name = "input.1";
    p.recognizer_output_name = "1333";
    p.recognizer_input_h = 112;
    p.recognizer_input_w = 112;
    p.recognizer_embedding_dim = 512;

    p.alignment_template = {
        {38.2946f, 51.6963f},
        {73.5318f, 51.5014f},
        {56.0252f, 71.7366f},
        {41.5493f, 92.3655f},
        {70.7299f, 92.2041f},
    };
    p.alignment_crop_h = 112;
    p.alignment_crop_w = 112;
    p.alignment_crop_size = 112;
    return p;
}

} // namespace

ModelProfile make_default_model_profile() {
    return build_default_profile();
}

VideoFacePipeline::VideoFacePipeline() = default;

VideoFacePipeline::~VideoFacePipeline() {
    if (d_detector_input_) cudaFree(d_detector_input_);
    if (d_surface_ptrs_) cudaFree(d_surface_ptrs_);
    if (d_pitches_) cudaFree(d_pitches_);
    if (d_widths_) cudaFree(d_widths_);
    if (d_heights_) cudaFree(d_heights_);
    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
}

bool VideoFacePipeline::init(int gpu_id,
                             const std::string& retinaface_engine_path,
                             const std::string& glintr100_engine_path,
                             std::string* error) {
    gpu_id_ = gpu_id;
    profile_ = build_default_profile();

    cudaError_t cuerr = cudaSetDevice(gpu_id_);
    if (cuerr != cudaSuccess) {
        if (error) *error = std::string("cudaSetDevice failed: ") + cudaGetErrorString(cuerr);
        return false;
    }

    cuerr = cudaStreamCreate(&stream_);
    if (cuerr != cudaSuccess) {
        if (error) *error = std::string("cudaStreamCreate failed: ") + cudaGetErrorString(cuerr);
        return false;
    }

    try {
        if (!allocate_detector_buffers()) {
            if (error) *error = "allocate_detector_buffers failed";
            return false;
        }

        retina_engine_ = std::make_unique<mergenvision::RetinaFaceEngine>(
            profile_, retinaface_engine_path, gpu_id_, stream_);
        if (!retina_engine_->init()) {
            if (error) *error = "RetinaFace engine init failed";
            return false;
        }

        retina_postproc_ = std::make_unique<mergenvision::RetinaFacePostproc>(
            profile_.detector_input_size,
            profile_.detector_max_candidates,
            gpu_id_,
            stream_);

        glint_engine_ = std::make_unique<mergenvision::GlintR100Engine>();
        if (!glint_engine_->load(gpu_id_, profile_, glintr100_engine_path, error)) {
            if (error && error->empty()) *error = "GlintR100 engine load failed";
            return false;
        }
    } catch (const std::exception& e) {
        if (error) *error = std::string("pipeline init exception: ") + e.what();
        return false;
    }

    initialized_ = true;
    return true;
}

bool VideoFacePipeline::allocate_detector_buffers() {
    // Support up to batch-32 smoke matrix; engine profile max is 64 but most
    // test cards run out of surface memory before that.
    const size_t max_batch = 32;
    detector_input_bytes_ = max_batch * 3 * profile_.detector_input_size *
                            profile_.detector_input_size * sizeof(float);

    CU_CHECK(cudaMalloc(&d_detector_input_, detector_input_bytes_));

    h_surface_ptrs_.resize(max_batch);
    h_pitches_.resize(max_batch);
    h_widths_.resize(max_batch);
    h_heights_.resize(max_batch);

    CU_CHECK(cudaMalloc(&d_surface_ptrs_, max_batch * sizeof(uint8_t*)));
    CU_CHECK(cudaMalloc(&d_pitches_, max_batch * sizeof(int)));
    CU_CHECK(cudaMalloc(&d_widths_, max_batch * sizeof(int)));
    CU_CHECK(cudaMalloc(&d_heights_, max_batch * sizeof(int)));

    h_uv_ptrs_.resize(max_batch);
    h_uv_pitches_.resize(max_batch);
    CU_CHECK(cudaMalloc(&d_uv_ptrs_, max_batch * sizeof(uint8_t*)));
    CU_CHECK(cudaMalloc(&d_uv_pitches_, max_batch * sizeof(int)));
    return true;
}

std::vector<FrameDetections> VideoFacePipeline::infer_detector_batch(
    const InferenceFrameBatch& batch) {
    if (!initialized_) {
        throw std::runtime_error("VideoFacePipeline not initialized");
    }

    const int n = static_cast<int>(batch.frames.size());
    if (n == 0) {
        return {};
    }
    if (n > 32) {
        throw std::runtime_error("detector batch size exceeds 32");
    }

    CU_CHECK(cudaSetDevice(gpu_id_));

    auto t_preprocess_start = std::chrono::steady_clock::now();

    std::vector<std::pair<int, int>> original_dims;
    original_dims.reserve(n);

    const int batch_format = batch.frames[0].device_view.format;
    for (int i = 1; i < n; ++i) {
        if (batch.frames[i].device_view.format != batch_format) {
            throw std::runtime_error("mixed surface color formats in detector batch");
        }
    }

    const bool rgba_mode = is_rgba_format(batch_format);
    const bool nv12_mode = is_nv12_family(batch_format);
    if (!rgba_mode && !nv12_mode) {
        throw std::runtime_error(
            std::string("VIDEO_PREPROCESS_COLOR_FORMAT_MISMATCH: ") +
            color_format_name(batch_format));
    }

    int color_mode = -1;
    if (nv12_mode) {
        color_mode = nv12_color_mode(batch_format);
        if (color_mode < 0) {
            throw std::runtime_error(
                std::string("unsupported NV12 variant: ") +
                color_format_name(batch_format));
        }
    }

    for (int i = 0; i < n; ++i) {
        const DeviceImageView& v = batch.frames[i].device_view;
        if (!v.data_ptr) {
            throw std::runtime_error("FrameEnvelope has null device_view data_ptr");
        }
        const uint8_t* base = static_cast<const uint8_t*>(v.data_ptr);
        // The actual sampled surface size is the NvBufSurface dimensions. The
        // original display dimensions are only used to map detector output back
        // to application-space coordinates.
        h_widths_[i] = static_cast<int>(v.width);
        h_heights_[i] = static_cast<int>(v.height);
        int orig_w = static_cast<int>(v.display_width ? v.display_width : v.width);
        int orig_h = static_cast<int>(v.display_height ? v.display_height : v.height);
        original_dims.emplace_back(orig_w, orig_h);

        if (rgba_mode) {
            h_surface_ptrs_[i] = base;
            h_pitches_[i] = static_cast<int>(v.pitch);
        } else {  // NV12/NV21
            if (v.num_planes < 2) {
                throw std::runtime_error("NV12 surface has fewer than 2 planes");
            }
            h_surface_ptrs_[i] = base + v.plane_offset[0];
            h_pitches_[i] = static_cast<int>(v.plane_pitch[0]);
            h_uv_ptrs_[i] = base + v.plane_offset[1];
            h_uv_pitches_[i] = static_cast<int>(v.plane_pitch[1]);
        }
    }

    CU_CHECK(cudaMemcpyAsync(d_surface_ptrs_, h_surface_ptrs_.data(),
                             n * sizeof(uint8_t*), cudaMemcpyHostToDevice, stream_));
    CU_CHECK(cudaMemcpyAsync(d_pitches_, h_pitches_.data(),
                             n * sizeof(int), cudaMemcpyHostToDevice, stream_));
    CU_CHECK(cudaMemcpyAsync(d_widths_, h_widths_.data(),
                             n * sizeof(int), cudaMemcpyHostToDevice, stream_));
    CU_CHECK(cudaMemcpyAsync(d_heights_, h_heights_.data(),
                             n * sizeof(int), cudaMemcpyHostToDevice, stream_));

    if (rgba_mode) {
        CU_CHECK(mergenvision_preprocess_detector_rgba_batch(
            d_surface_ptrs_, d_pitches_, d_widths_, d_heights_,
            n, d_detector_input_, stream_));
    } else {
        CU_CHECK(cudaMemcpyAsync(d_uv_ptrs_, h_uv_ptrs_.data(),
                                 n * sizeof(uint8_t*), cudaMemcpyHostToDevice, stream_));
        CU_CHECK(cudaMemcpyAsync(d_uv_pitches_, h_uv_pitches_.data(),
                                 n * sizeof(int), cudaMemcpyHostToDevice, stream_));
        CU_CHECK(mergenvision_preprocess_detector_nv12_batch(
            d_surface_ptrs_, d_uv_ptrs_, d_pitches_, d_uv_pitches_,
            d_widths_, d_heights_, n, color_mode, d_detector_input_, stream_));
    }

    auto t_preprocess_end = std::chrono::steady_clock::now();
    auto t_engine_start = std::chrono::steady_clock::now();

    const float* d_loc = nullptr;
    const float* d_conf = nullptr;
    const float* d_landms = nullptr;
    int num_anchors = 0;
    if (!retina_engine_->infer(d_detector_input_, n, &d_loc, &d_conf, &d_landms, &num_anchors)) {
        throw std::runtime_error("RetinaFace inference failed");
    }
    auto t_engine_end = std::chrono::steady_clock::now();
    auto t_postproc_start = std::chrono::steady_clock::now();

    std::vector<std::vector<mergenvision::FaceDetection>> raw_per_frame =
        retina_postproc_->processBatch(
            d_loc, d_conf, d_landms, num_anchors, n, original_dims,
            profile_.detector_conf_threshold,
            profile_.detector_nms_threshold);
    auto t_postproc_end = std::chrono::steady_clock::now();
    auto t_mapping_start = std::chrono::steady_clock::now();

    std::vector<FrameDetections> result;
    result.reserve(n);

    for (int i = 0; i < n; ++i) {
        FrameDetections fd;
        fd.frame = batch.frames[i];

        auto& raw_dets = raw_per_frame[i];
        // NaN rejection and stable sort.
        raw_dets.erase(
            std::remove_if(
                raw_dets.begin(), raw_dets.end(),
                [](const mergenvision::FaceDetection& d) {
                    return !std::isfinite(d.x1) || !std::isfinite(d.y1) ||
                           !std::isfinite(d.x2) || !std::isfinite(d.y2) ||
                           !std::isfinite(d.score);
                }),
            raw_dets.end());
        std::stable_sort(
            raw_dets.begin(), raw_dets.end(),
            [](const mergenvision::FaceDetection& a, const mergenvision::FaceDetection& b) {
                if (a.x1 != b.x1) return a.x1 < b.x1;
                if (a.y1 != b.y1) return a.y1 < b.y1;
                if (a.x2 != b.x2) return a.x2 < b.x2;
                if (a.y2 != b.y2) return a.y2 < b.y2;
                return a.score > b.score;
            });

        uint32_t ordinal = 0;
        for (const auto& rd : raw_dets) {
            FaceDetection face;
            face.frame = fd.frame;
            face.detection_ordinal = ordinal++;
            face.observation_id = std::string() + ":" +
                                  std::to_string(fd.frame.presentation_index) + ":" +
                                  std::to_string(face.detection_ordinal);
            face.bbox = {rd.x1, rd.y1, rd.x2, rd.y2};
            std::memcpy(face.landmarks.data(), rd.landmarks, 10 * sizeof(float));
            face.detector_score = rd.score;
            face.quality_score = rd.score;
            face.tracking_eligible = true;
            // Recognition eligibility will be decided by a later quality gate.
            face.recognition_eligible = false;
            face.rejection_code = "recognition_not_yet_implemented";
            face.model_version = profile_.model_version;
            face.preprocess_version = profile_.preprocess_version;
            fd.detections.push_back(std::move(face));
        }
        result.push_back(std::move(fd));
    }

    auto t_mapping_end = std::chrono::steady_clock::now();
    auto us = [](auto a, auto b) {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
    };
    metrics_.preprocess_us += us(t_preprocess_start, t_preprocess_end);
    metrics_.engine_enqueue_us += us(t_engine_start, t_engine_end);
    metrics_.postproc_us += us(t_postproc_start, t_postproc_end);
    metrics_.mapping_us += us(t_mapping_start, t_mapping_end);
    ++metrics_.total_calls;

    return result;
}

std::vector<EmbeddingResult> VideoFacePipeline::infer_recognition_batch(
    const std::vector<RecognitionCropRef>& /*crops*/) {
    // Placeholder until M5.2 recognition integration.
    return {};
}

} // namespace mergenvision::video
