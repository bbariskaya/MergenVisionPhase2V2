#pragma once

#include "mv/video/frame_identity.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mergenvision::video {

class PresentationOrderViolation : public std::runtime_error {
public:
    explicit PresentationOrderViolation(const std::string& what)
        : std::runtime_error(what) {}
};

struct InferenceFrameBatch {
    uint64_t batch_sequence = 0;
    std::vector<FrameEnvelope> frames;
};

struct BatchAssemblerDiagnostics {
    uint64_t duplicate_pts_count = 0;
    uint64_t gap_count = 0;
    uint64_t nonmonotonic_pts_count = 0;
};

// Authoritative temporal batch assembler.
//
// Responsibilities:
//   * flatten irregular nvstreammux GstBuffer outputs
//   * enforce canonical presentation order (PTS, then nvds_frame_num, then batch_id)
//   * form exact detector inference batches of up to |max_batch_size| frames
//   * preserve retained GstBuffer ownership across batch boundaries
//   * flush a partial batch at EOS
//
// The assembler never recreates frame identity. It only assigns
// |inference_batch_sequence| and |position_in_inference_batch| fields on the
// emitted frames and leaves all other identity fields untouched.
class TemporalFrameBatchAssembler {
public:
    explicit TemporalFrameBatchAssembler(size_t max_batch_size,
                                         bool sampling_enabled = false);

    // Consume one mux output buffer's worth of frames (may be empty).
    // Returns zero or more complete inference batches.
    std::vector<InferenceFrameBatch> push(std::vector<FrameEnvelope> mux_frames);

    // Return any remaining frames as the final partial inference batch.
    std::optional<InferenceFrameBatch> flush_eos();

    // Release all retained frames without emitting them.
    void cancel();

    BatchAssemblerDiagnostics diagnostics() const { return diagnostics_; }

    size_t pending_count() const { return pending_.size(); }

private:
    InferenceFrameBatch emit_one_batch(size_t count);

    size_t max_batch_size_;
    bool sampling_enabled_ = false;
    uint64_t next_inference_sequence_ = 0;
    std::vector<FrameEnvelope> pending_;
    int64_t previous_pts_ns_ = -1;
    uint64_t previous_presentation_index_ = static_cast<uint64_t>(-1);
    BatchAssemblerDiagnostics diagnostics_{};
};

}  // namespace mergenvision::video
