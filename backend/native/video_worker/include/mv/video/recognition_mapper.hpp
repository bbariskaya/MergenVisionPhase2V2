#pragma once

#include "mv/video/frame_identity.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mergenvision::video {

// Placeholder for a device-resident aligned crop. The real implementation will
// carry a CUDA pointer/pitch; the metadata contract only needs identity here.
struct DeviceAlignedCrop {
    void* data_ptr = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t pitch = 0;
};

struct RecognitionCropRef {
    uint64_t presentation_index = 0;
    uint32_t detection_ordinal = 0;
    uint32_t detector_batch_position = 0;
    DeviceAlignedCrop crop;
    std::array<float, 10> landmarks{};  // five source-space landmarks x0,y0..x4,y4
};

struct EmbeddingResult {
    uint64_t presentation_index = 0;
    uint32_t detection_ordinal = 0;
    std::vector<float> embedding;
    std::string model_version;
    std::string preprocess_version;
};

// Deterministic ordering required by the contract.
inline bool operator<(const RecognitionCropRef& a, const RecognitionCropRef& b) noexcept {
    if (a.presentation_index != b.presentation_index) {
        return a.presentation_index < b.presentation_index;
    }
    return a.detection_ordinal < b.detection_ordinal;
}

// Chunk a flat deterministic crop list into recognizer batches of at most
// |max_batch_size| crops. Empty input yields zero chunks.
inline std::vector<std::vector<RecognitionCropRef>> chunk_recognition_crops(
    const std::vector<RecognitionCropRef>& crops, size_t max_batch_size = 32) {
    if (max_batch_size == 0) {
        throw std::invalid_argument("max_batch_size must be > 0");
    }
    std::vector<std::vector<RecognitionCropRef>> chunks;
    for (size_t i = 0; i < crops.size(); i += max_batch_size) {
        size_t end = std::min(i + max_batch_size, crops.size());
        chunks.emplace_back(crops.begin() + static_cast<std::ptrdiff_t>(i),
                            crops.begin() + static_cast<std::ptrdiff_t>(end));
    }
    return chunks;
}

// Map recognizer output embeddings back to the exact crop references by chunk
// index and in-chunk position. The outer |chunk_embeddings| vector must have
// the same number of inner vectors as |chunks|, and each inner vector must
// have the same length as the corresponding chunk.
inline std::vector<EmbeddingResult> map_recognition_embeddings(
    const std::vector<std::vector<RecognitionCropRef>>& chunks,
    const std::vector<std::vector<float>>& chunk_embeddings,
    const std::string& model_version = "glintr100_v1",
    const std::string& preprocess_version = "align_5point_v1") {
    if (chunks.size() != chunk_embeddings.size()) {
        throw std::invalid_argument("chunk count and embedding chunk count mismatch");
    }

    std::vector<EmbeddingResult> results;
    for (size_t c = 0; c < chunks.size(); ++c) {
        const auto& chunk = chunks[c];
        const auto& embs = chunk_embeddings[c];
        if (chunk.size() != embs.size() / 512) {
            throw std::invalid_argument("chunk size and embedding chunk size mismatch");
        }
        for (size_t i = 0; i < chunk.size(); ++i) {
            EmbeddingResult er;
            er.presentation_index = chunk[i].presentation_index;
            er.detection_ordinal = chunk[i].detection_ordinal;
            er.embedding.assign(embs.begin() + static_cast<std::ptrdiff_t>(i * 512),
                                embs.begin() + static_cast<std::ptrdiff_t>((i + 1) * 512));
            er.model_version = model_version;
            er.preprocess_version = preprocess_version;
            results.push_back(std::move(er));
        }
    }
    return results;
}

}  // namespace mergenvision::video
