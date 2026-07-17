#include "mv/video/batch_assembler.hpp"

#include <algorithm>
#include <sstream>

namespace mergenvision::video {

namespace {

bool presentation_order_less(const FrameEnvelope& a, const FrameEnvelope& b) noexcept {
    if (a.pts_ns != b.pts_ns) return a.pts_ns < b.pts_ns;
    if (a.nvds_frame_num != b.nvds_frame_num) return a.nvds_frame_num < b.nvds_frame_num;
    return a.position_in_mux_batch < b.position_in_mux_batch;
}

}  // namespace

TemporalFrameBatchAssembler::TemporalFrameBatchAssembler(size_t max_batch_size,
                                                         bool sampling_enabled)
    : max_batch_size_(max_batch_size), sampling_enabled_(sampling_enabled) {
    if (max_batch_size_ == 0) {
        throw std::invalid_argument("max_batch_size must be > 0");
    }
}

std::vector<InferenceFrameBatch> TemporalFrameBatchAssembler::push(
    std::vector<FrameEnvelope> mux_frames) {
    std::vector<InferenceFrameBatch> complete;

    if (!mux_frames.empty()) {
        // Canonical ordering inside this mux buffer: PTS ascending, then
        // nvds_frame_num, then surface slot. We do not assume the mux output is
        // already in presentation order.
        std::sort(mux_frames.begin(), mux_frames.end(), presentation_order_less);

        for (auto& env : mux_frames) {
            if (env.pts_ns < 0) {
                env.pts_derived = true;
            }

            if (previous_pts_ns_ >= 0 && env.pts_ns >= 0) {
                if (env.pts_ns < previous_pts_ns_) {
                    std::ostringstream oss;
                    oss << "VIDEO_PRESENTATION_ORDER_VIOLATION: pts regressed from "
                        << previous_pts_ns_ << " ns to " << env.pts_ns << " ns";
                    throw PresentationOrderViolation(oss.str());
                }
                if (env.pts_ns == previous_pts_ns_) {
                    ++diagnostics_.duplicate_pts_count;
                }
            }

            if (!sampling_enabled_ &&
                previous_presentation_index_ != static_cast<uint64_t>(-1)) {
                if (env.presentation_index != previous_presentation_index_ + 1) {
                    ++diagnostics_.gap_count;
                }
            }

            previous_pts_ns_ = env.pts_ns;
            previous_presentation_index_ = env.presentation_index;
            pending_.push_back(std::move(env));

            while (pending_.size() >= max_batch_size_) {
                complete.push_back(emit_one_batch(max_batch_size_));
            }
        }
    }

    return complete;
}

std::optional<InferenceFrameBatch> TemporalFrameBatchAssembler::flush_eos() {
    if (pending_.empty()) {
        return std::nullopt;
    }
    return emit_one_batch(pending_.size());
}

void TemporalFrameBatchAssembler::cancel() {
    pending_.clear();
}

InferenceFrameBatch TemporalFrameBatchAssembler::emit_one_batch(size_t count) {
    if (count == 0 || count > pending_.size()) {
        throw std::logic_error("emit_one_batch called with invalid count");
    }

    InferenceFrameBatch batch;
    batch.batch_sequence = next_inference_sequence_++;
    batch.frames.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        FrameEnvelope env = std::move(pending_[i]);
        env.inference_batch_sequence = static_cast<uint32_t>(batch.batch_sequence);
        env.position_in_inference_batch = static_cast<uint32_t>(i);
        batch.frames.push_back(std::move(env));
    }

    pending_.erase(pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(count));
    return batch;
}

}  // namespace mergenvision::video
