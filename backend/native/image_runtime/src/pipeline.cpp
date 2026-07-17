#include "pipeline.h"
#include "util.h"
#include "webp_encoder.h"
#include "mergenvision_kernels.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

namespace mergenvision {

namespace {

constexpr int kDetectorSize = 640;
constexpr int kCropSize = 112;
constexpr int kEmbeddingDim = 512;
constexpr float kConfThreshold = 0.5f;
constexpr float kNmsThreshold = 0.4f;
constexpr int kMaxCandidates = 300;
constexpr float kWebpQuality = 95.0f;
constexpr float kL2Epsilon = 1e-12f;

} // namespace

ExecutionSlot::ExecutionSlot(int device_id,
                             const std::string& retinaface_engine_path,
                             const std::string& glintr100_engine_path,
                             std::string* error)
    : device_id_(device_id) {
    CU_CHECK(cudaSetDevice(device_id_));
    CU_CHECK(cudaStreamCreate(&stream_));

    if (!decoder_.init(error)) {
        if (error) *error = "JpegDecoder init failed: " + *error;
        return;
    }

    retina_engine_ = std::make_unique<RetinaFaceEngine>(retinaface_engine_path, device_id_, stream_);
    if (!retina_engine_->init()) {
        if (error) *error = "RetinaFace engine init failed";
        return;
    }

    retina_postproc_ = std::make_unique<RetinaFacePostproc>(
        kDetectorSize, kMaxCandidates, device_id_, stream_);

    glint_engine_ = std::make_unique<GlintR100Engine>();
    if (!glint_engine_->load(device_id_, glintr100_engine_path, error)) {
        if (error) *error = "GlintR100 engine load failed: " + (error ? *error : "");
        return;
    }

    max_faces_ = glint_engine_->max_batch();
    if (max_faces_ < 1) {
        if (error) *error = "GlintR100 engine max batch < 1";
        return;
    }

    if (!ensure_buffers(max_faces_, error)) {
        return;
    }
}

ExecutionSlot::~ExecutionSlot() {
    cudaSetDevice(device_id_);
    CU_CHECK(cudaStreamSynchronize(stream_));
    CU_CHECK(cudaFree(d_detector_input_));
    CU_CHECK(cudaFree(d_aligned_crops_));
    CU_CHECK(cudaFree(d_face_landmarks_));
    CU_CHECK(cudaFree(d_face_matrices_));
    CU_CHECK(cudaFree(d_kernel_status_));
    CU_CHECK(cudaFreeHost(h_embeddings_));
    CU_CHECK(cudaFreeHost(h_aligned_crops_));
    cudaStreamDestroy(stream_);
}

bool ExecutionSlot::ensure_buffers(int max_faces, std::string* error) {
    if (max_faces <= 0) {
        if (error) *error = "ensure_buffers: max_faces <= 0";
        return false;
    }

    detector_input_bytes_ = static_cast<size_t>(3) * kDetectorSize * kDetectorSize * sizeof(float);

    size_t aligned_bytes = static_cast<size_t>(max_faces) * 3 * kCropSize * kCropSize * sizeof(float);
    size_t landmarks_bytes = static_cast<size_t>(max_faces) * 10 * sizeof(float);
    size_t matrices_bytes = static_cast<size_t>(max_faces) * 6 * sizeof(float);
    size_t embeddings_bytes = static_cast<size_t>(max_faces) * kEmbeddingDim * sizeof(float);

    CU_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_detector_input_), detector_input_bytes_));
    CU_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aligned_crops_), aligned_bytes));
    CU_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_face_landmarks_), landmarks_bytes));
    CU_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_face_matrices_), matrices_bytes));
    CU_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_kernel_status_), sizeof(int)));

    CU_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_embeddings_), embeddings_bytes));
    CU_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_aligned_crops_), aligned_bytes));

    return true;
}

bool ExecutionSlot::infer_jpeg(const void* jpeg_data,
                               std::size_t jpeg_size,
                               InferenceResult* result,
                               std::string* error) {
    if (!result) {
        if (error) *error = "result is null";
        return false;
    }
    result->detections.clear();

    CU_CHECK(cudaSetDevice(device_id_));

    int img_w = 0;
    int img_h = 0;
    unsigned char* d_rgb = nullptr;
    if (!decoder_.decode(jpeg_data, jpeg_size, &img_w, &img_h, &d_rgb, stream_, error)) {
        return false;
    }

    result->image_width = img_w;
    result->image_height = img_h;

    CU_CHECK(cudaMemsetAsync(d_kernel_status_, 0, sizeof(int), stream_));

    // Preprocess to BGR NCHW 640x640 for RetinaFace.
    CU_CHECK(static_cast<cudaError_t>(
        mergenvision_preprocess_retinaface(d_rgb, img_h, img_w, d_detector_input_, stream_)));

    // Run detector.
    const float* d_loc = nullptr;
    const float* d_conf = nullptr;
    const float* d_landms = nullptr;
    int num_anchors = 0;
    if (!retina_engine_->infer(d_detector_input_, 1, &d_loc, &d_conf, &d_landms, &num_anchors)) {
        CU_CHECK(cudaFree(d_rgb));
        if (error) *error = "RetinaFace inference failed";
        return false;
    }

    // Decode and NMS on GPU.
    std::vector<FaceDetection> detections = retina_postproc_->processFrame(
        d_loc, d_conf, d_landms, num_anchors, img_w, img_h, kConfThreshold, kNmsThreshold);

    if (detections.empty()) {
        CU_CHECK(cudaFree(d_rgb));
        return true;
    }

    if (static_cast<int>(detections.size()) > max_faces_) {
        CU_CHECK(cudaFree(d_rgb));
        if (error) {
            std::ostringstream oss;
            oss << "too many faces: " << detections.size() << " (max " << max_faces_ << ")";
            *error = oss.str();
        }
        return false;
    }

    int n_faces = static_cast<int>(detections.size());

    // Build per-face landmark array on host and copy to device for similarity transform.
    std::vector<float> host_landmarks(n_faces * 10);
    for (int i = 0; i < n_faces; ++i) {
        for (int k = 0; k < 10; ++k) {
            host_landmarks[i * 10 + k] = detections[i].landmarks[k];
        }
    }
    CU_CHECK(cudaMemcpyAsync(d_face_landmarks_, host_landmarks.data(),
                             host_landmarks.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream_));

    // Compute M = [[a,-b,tx],[b,a,ty]] for each face.
    CU_CHECK(static_cast<cudaError_t>(
        mergenvision_similarity_transform(d_face_landmarks_, d_face_matrices_,
                                          n_faces, kCropSize, d_kernel_status_, stream_)));

    // Warp-align all faces from original RGB image.
    CU_CHECK(static_cast<cudaError_t>(
        mergenvision_warp_align(d_rgb, img_h, img_w, d_face_matrices_,
                                n_faces, d_aligned_crops_, stream_)));

    // Copy aligned crops into recognizer input buffer.
    size_t aligned_bytes = static_cast<size_t>(n_faces) * 3 * kCropSize * kCropSize * sizeof(float);
    CU_CHECK(cudaMemcpyAsync(glint_engine_->input_buffer(), d_aligned_crops_, aligned_bytes,
                             cudaMemcpyDeviceToDevice, stream_));

    // Recognize.
    if (!glint_engine_->enqueue(n_faces, stream_, error)) {
        CU_CHECK(cudaFree(d_rgb));
        return false;
    }

    // L2-normalize embeddings in place.
    CU_CHECK(cudaMemsetAsync(d_kernel_status_, 0, sizeof(int), stream_));
    CU_CHECK(static_cast<cudaError_t>(
        mergenvision_l2_normalize(glint_engine_->output_buffer(), glint_engine_->output_buffer(),
                                  n_faces, kEmbeddingDim, kL2Epsilon,
                                  d_kernel_status_, stream_)));

    // Copy embeddings and aligned crops back to host.
    size_t embeddings_bytes = static_cast<size_t>(n_faces) * kEmbeddingDim * sizeof(float);
    CU_CHECK(cudaMemcpyAsync(h_embeddings_, glint_engine_->output_buffer(), embeddings_bytes,
                             cudaMemcpyDeviceToHost, stream_));
    CU_CHECK(cudaMemcpyAsync(h_aligned_crops_, d_aligned_crops_, aligned_bytes,
                             cudaMemcpyDeviceToHost, stream_));

    // Wait for all work and status flags.
    int host_status = 0;
    CU_CHECK(cudaMemcpyAsync(&host_status, d_kernel_status_, sizeof(int),
                             cudaMemcpyDeviceToHost, stream_));
    CU_CHECK(cudaStreamSynchronize(stream_));

    if (host_status != 0) {
        CU_CHECK(cudaFree(d_rgb));
        if (error) *error = "kernel reported non-finite/zero norm during alignment or normalization";
        return false;
    }

    CU_CHECK(cudaFree(d_rgb));
    d_rgb = nullptr;

    // Build CPU result.
    result->detections.reserve(n_faces);
    for (int i = 0; i < n_faces; ++i) {
        FaceObservation obs;
        obs.detection_index = i;
        const FaceDetection& d = detections[i];
        obs.x = d.x1;
        obs.y = d.y1;
        obs.width = d.x2 - d.x1;
        obs.height = d.y2 - d.y1;
        obs.detector_confidence = d.score;
        obs.landmarks5.assign(d.landmarks, d.landmarks + 10);

        obs.embedding.resize(kEmbeddingDim);
        std::memcpy(obs.embedding.data(),
                    h_embeddings_ + i * kEmbeddingDim,
                    kEmbeddingDim * sizeof(float));

        size_t crop_ofs = static_cast<size_t>(i) * 3 * kCropSize * kCropSize;
        obs.aligned_crop_bytes = encode_aligned_crop_webp(
            h_aligned_crops_ + crop_ofs, kCropSize, kCropSize, kWebpQuality);
        if (obs.aligned_crop_bytes.empty()) {
            if (error) *error = "WebP crop encoding failed";
            return false;
        }

        result->detections.push_back(std::move(obs));
    }

    return true;
}

} // namespace mergenvision
