#pragma once

#include "mv/video/frame_identity.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace mergenvision::video {

struct BBox {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;

    bool operator==(const BBox& other) const noexcept {
        return x1 == other.x1 && y1 == other.y1 && x2 == other.x2 && y2 == other.y2;
    }
};

// Per-frame face detection in original display-space coordinates.
struct FaceDetection {
    FrameIdentity frame;
    uint32_t detection_ordinal = 0;
    std::string observation_id;
    BBox bbox;
    std::array<float, 10> landmarks{};
    float detector_score = 0.0f;
    float quality_score = 0.0f;
    bool tracking_eligible = false;
    bool recognition_eligible = false;
    std::string rejection_code;
    std::vector<float> embedding;
    std::string model_version;
    std::string preprocess_version;

    // Populated by the tracker in the native worker so the Python side can
    // correlate frame detections with the raw-track template/crop artifacts.
    std::string raw_track_key;
};

struct FrameDetections {
    FrameIdentity frame;
    std::vector<FaceDetection> detections;
};

// Raw detection as emitted by the TensorRT RetinaFace output, still indexed by
// its position inside the detector inference batch. The mapping helper is
// responsible for reversing the per-frame preprocessing transform and tagging
// each detection with the correct FrameIdentity.
struct RawDetection {
    uint32_t position_in_inference_batch = 0;
    BBox bbox;
    float detector_score = 0.0f;
    std::array<float, 10> landmarks{};
    bool recognition_eligible = false;
};

// Map raw detector output back to the correct input frames. The returned vector
// has exactly one entry per input frame in the same order as the inference
// batch (presentation index ascending). Even frames with zero detections are
// represented with an empty |detections| vector — frame identity is never lost.
inline std::vector<FrameDetections> map_detector_output_to_frames(
    const InferenceFrameBatch& batch,
    const std::vector<RawDetection>& raw_detections,
    const std::string& job_id) {
    std::vector<FrameDetections> result;
    result.reserve(batch.frames.size());

    // Group raw detections by their source inference-batch position.
    std::vector<std::vector<RawDetection>> by_position(batch.frames.size());
    for (const auto& det : raw_detections) {
        if (det.position_in_inference_batch >= by_position.size()) {
            continue;
        }
        by_position[det.position_in_inference_batch].push_back(det);
    }

    // Map to frames, stable-sort inside each frame, assign ordinal and
    // deterministic observation id.
    for (size_t i = 0; i < batch.frames.size(); ++i) {
        const FrameIdentity& fid = batch.frames[i];
        FrameDetections fd;
        fd.frame = fid;

        auto& dets = by_position[i];
        // NaN rejection.
        dets.erase(
            std::remove_if(
                dets.begin(), dets.end(),
                [](const RawDetection& d) {
                    return !std::isfinite(d.bbox.x1) || !std::isfinite(d.bbox.y1) ||
                           !std::isfinite(d.bbox.x2) || !std::isfinite(d.bbox.y2) ||
                           !std::isfinite(d.detector_score);
                }),
            dets.end());

        std::stable_sort(
            dets.begin(), dets.end(),
            [](const RawDetection& a, const RawDetection& b) {
                if (a.bbox.x1 != b.bbox.x1) return a.bbox.x1 < b.bbox.x1;
                if (a.bbox.y1 != b.bbox.y1) return a.bbox.y1 < b.bbox.y1;
                if (a.bbox.x2 != b.bbox.x2) return a.bbox.x2 < b.bbox.x2;
                if (a.bbox.y2 != b.bbox.y2) return a.bbox.y2 < b.bbox.y2;
                return a.detector_score > b.detector_score;
            });

        uint32_t ordinal = 0;
        for (const auto& det : dets) {
            FaceDetection face;
            face.frame = fid;
            face.detection_ordinal = ordinal++;
            face.observation_id = job_id + ":" + std::to_string(fid.presentation_index) +
                                  ":" + std::to_string(face.detection_ordinal);
            face.bbox = det.bbox;
            face.landmarks = det.landmarks;
            face.detector_score = det.detector_score;
            face.recognition_eligible = det.recognition_eligible;
            face.tracking_eligible = true;
            fd.detections.push_back(std::move(face));
        }
        result.push_back(std::move(fd));
    }

    return result;
}

}  // namespace mergenvision::video
