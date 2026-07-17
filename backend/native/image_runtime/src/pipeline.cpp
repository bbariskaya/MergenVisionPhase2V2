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

constexpr float kWebpQuality = 95.0f;
constexpr float kL2Epsilon = 1e-12f;

} // namespace

ExecutionSlot::ExecutionSlot(const ModelProfile& profile,
                             int device_id,
                             const std::string& retinaface_engine_path,
                             const std::string& glintr100_engine_path,
                             std::string* error)
    : profile_(profile), device_id_(device_id) {
    CU_CHECK(cudaSetDevice(device_id_));
    CU_CHECK(cudaStreamCreate(&stream_));

    if (!decoder_.init(error)) {
        if (error) *error = "JpegDecoder init failed: " + *error;
        state_ = State::unavailable;
        return;
    }

    retina_engine_ = std::make_unique<RetinaFaceEngine>(profile_, retinaface_engine_path, device_id_, stream_);
    if (!retina_engine_->init()) {
        if (error) *error = "RetinaFace engine init failed";
        state_ = State::unavailable;
        return;
    }

    retina_postproc_ = std::make_unique<RetinaFacePostproc>(
        profile_.detector_input_size, profile_.detector_max_candidates, device_id_, stream_);

    glint_engine_ = std::make_unique<GlintR100Engine>();
    if (!glint_engine_->load(device_id_, profile_, glintr100_engine_path, error)) {
        if (error) *error = "GlintR100 engine load failed: " + (error ? *error : "");
        state_ = State::unavailable;
        return;
    }

    max_faces_ = glint_engine_->max_batch();
    if (max_faces_ < 1) {
        if (error) *error = "GlintR100 engine max batch < 1";
        state_ = State::unavailable;
        return;
    }

    if (!ensure_buffers(max_faces_, error)) {
        state_ = State::unavailable;
        return;
    }

    state_ = State::initialized;
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

    detector_input_bytes_ = static_cast<size_t>(3) * profile_.detector_input_size * profile_.detector_input_size * sizeof(float);

    size_t aligned_bytes = static_cast<size_t>(max_faces) * 3 * profile_.alignment_crop_size * profile_.alignment_crop_size * sizeof(float);
    size_t landmarks_bytes = static_cast<size_t>(max_faces) * 10 * sizeof(float);
    size_t matrices_bytes = static_cast<size_t>(max_faces) * 6 * sizeof(float);
    size_t embeddings_bytes = static_cast<size_t>(max_faces) * profile_.recognizer_embedding_dim * sizeof(float);

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

    auto d_rgb_deleter = [](unsigned char* p) { if (p) cudaFree(p); };
    std::unique_ptr<unsigned char, decltype(d_rgb_deleter)> d_rgb_guard(d_rgb, d_rgb_deleter);

    result->image_width = img_w;
    result->image_height = img_h;

    CU_CHECK(static_cast<cudaError_t>(
        mergenvision_preprocess_retinaface(d_rgb, img_h, img_w, d_detector_input_, stream_)));

    const float* d_loc = nullptr;
    const float* d_conf = nullptr;
    const float* d_landms = nullptr;
    int num_anchors = 0;
    if (!retina_engine_->infer(d_detector_input_, 1, &d_loc, &d_conf, &d_landms, &num_anchors)) {
        if (error) *error = "RetinaFace inference failed";
        return false;
    }

    std::vector<FaceDetection> detections = retina_postproc_->processFrame(
        d_loc, d_conf, d_landms, num_anchors, img_w, img_h,
        profile_.detector_conf_threshold, profile_.detector_nms_threshold);

    if (detections.empty()) {
        return true;
    }

    int n_faces = static_cast<int>(detections.size());
    result->detections.reserve(n_faces);

    // Process recognition in deterministic chunks bounded by the recognizer's
    // max batch size (32). Detection order is preserved.
    for (int chunk_start = 0; chunk_start < n_faces; chunk_start += max_faces_) {
        int chunk_end = std::min(chunk_start + max_faces_, n_faces);
        int chunk_size = chunk_end - chunk_start;

        std::vector<float> host_landmarks(static_cast<size_t>(chunk_size) * 10);
        for (int i = 0; i < chunk_size; ++i) {
            const FaceDetection& d = detections[chunk_start + i];
            for (int k = 0; k < 10; ++k) {
                host_landmarks[static_cast<size_t>(i) * 10 + k] = d.landmarks[k];
            }
        }
        CU_CHECK(cudaMemcpyAsync(d_face_landmarks_, host_landmarks.data(),
                                 host_landmarks.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, stream_));

        // Alignment: status flag is isolated and read before L2 normalization.
        CU_CHECK(cudaMemsetAsync(d_kernel_status_, 0, sizeof(int), stream_));
        CU_CHECK(static_cast<cudaError_t>(
            mergenvision_similarity_transform(d_face_landmarks_, d_face_matrices_,
                                              chunk_size, profile_.alignment_crop_size, d_kernel_status_, stream_)));

        int alignment_status = 0;
        CU_CHECK(cudaMemcpyAsync(&alignment_status, d_kernel_status_, sizeof(int),
                                 cudaMemcpyDeviceToHost, stream_));
        CU_CHECK(cudaStreamSynchronize(stream_));
        if (alignment_status != 0) {
            if (error) *error = "degenerate similarity transform rejected";
            return false;
        }

        CU_CHECK(static_cast<cudaError_t>(
            mergenvision_warp_align(d_rgb, img_h, img_w, d_face_matrices_,
                                    chunk_size, d_aligned_crops_, stream_)));

        size_t aligned_bytes = static_cast<size_t>(chunk_size) * 3 * profile_.alignment_crop_size * profile_.alignment_crop_size * sizeof(float);
        CU_CHECK(cudaMemcpyAsync(glint_engine_->input_buffer(), d_aligned_crops_, aligned_bytes,
                                 cudaMemcpyDeviceToDevice, stream_));

        if (!glint_engine_->enqueue(chunk_size, stream_, error)) {
            return false;
        }

        CU_CHECK(cudaMemsetAsync(d_kernel_status_, 0, sizeof(int), stream_));
        CU_CHECK(static_cast<cudaError_t>(
            mergenvision_l2_normalize(glint_engine_->output_buffer(), glint_engine_->output_buffer(),
                                      chunk_size, profile_.recognizer_embedding_dim, kL2Epsilon,
                                      d_kernel_status_, stream_)));

        size_t embeddings_bytes = static_cast<size_t>(chunk_size) * profile_.recognizer_embedding_dim * sizeof(float);
        CU_CHECK(cudaMemcpyAsync(h_embeddings_, glint_engine_->output_buffer(), embeddings_bytes,
                                 cudaMemcpyDeviceToHost, stream_));
        CU_CHECK(cudaMemcpyAsync(h_aligned_crops_, d_aligned_crops_, aligned_bytes,
                                 cudaMemcpyDeviceToHost, stream_));

        int l2_status = 0;
        CU_CHECK(cudaMemcpyAsync(&l2_status, d_kernel_status_, sizeof(int),
                                 cudaMemcpyDeviceToHost, stream_));
        CU_CHECK(cudaStreamSynchronize(stream_));
        if (l2_status != 0) {
            if (error) *error = "L2 normalization reported non-finite or zero norm";
            return false;
        }

        for (int i = 0; i < chunk_size; ++i) {
            FaceObservation obs;
            obs.detection_index = chunk_start + i;
            const FaceDetection& d = detections[chunk_start + i];
            obs.x = d.x1;
            obs.y = d.y1;
            obs.width = d.x2 - d.x1;
            obs.height = d.y2 - d.y1;
            obs.detector_confidence = d.score;
            obs.landmarks5.assign(d.landmarks, d.landmarks + 10);

            obs.embedding.resize(profile_.recognizer_embedding_dim);
            std::memcpy(obs.embedding.data(),
                        h_embeddings_ + static_cast<size_t>(i) * profile_.recognizer_embedding_dim,
                        profile_.recognizer_embedding_dim * sizeof(float));

            size_t crop_ofs = static_cast<size_t>(i) * 3 * profile_.alignment_crop_size * profile_.alignment_crop_size;
            obs.aligned_crop_bytes = encode_aligned_crop_webp(
                h_aligned_crops_ + crop_ofs, profile_.alignment_crop_size, profile_.alignment_crop_size, kWebpQuality);
            if (obs.aligned_crop_bytes.empty()) {
                if (error) *error = "WebP crop encoding failed";
                return false;
            }

            result->detections.push_back(std::move(obs));
        }
    }

    return true;
}

} // namespace mergenvision
