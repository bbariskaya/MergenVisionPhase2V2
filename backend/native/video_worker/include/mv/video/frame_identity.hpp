#pragma once

#include <cstdint>
#include <memory>

namespace mergenvision::video {

// Lightweight descriptor of a device-resident image surface. The actual
// pixel data remains on the GPU; this view is only valid while the owning
// retained buffer handle is alive.
struct DeviceImageView {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t display_width = 0;
    uint32_t display_height = 0;
    uint32_t pitch = 0;        // RGBA: line pitch in bytes
    int format = 0;            // NvBufSurfaceColorFormat value
    void* data_ptr = nullptr;  // base device pointer to buffer
    int surface_index = -1;

    // NV12/NV21 multi-plane layout. For RGBA surfaces leave these at zero
    // and use |pitch| / |data_ptr| directly.
    uint32_t num_planes = 0;
    uint32_t plane_offset[2] = {};
    uint32_t plane_pitch[2] = {};
};

// Abstract ownership handle that keeps the source GstBuffer/GstSample alive
// until all GPU work referencing its device view has completed.
class RetainedBufferHandle {
public:
    virtual ~RetainedBufferHandle() = default;
    virtual void release() = 0;
    virtual bool is_released() const = 0;
};

// Canonical identity semantics for a decoded video frame. These fields are
// independent of any hardware-specific counters and are the authoritative
// source for presentation order, timeline identity and observation identity.
struct FrameIdentity {
    uint64_t presentation_index = 0;
    uint64_t decoded_sequence = 0;
    uint64_t sampled_sequence = 0;
    uint64_t mux_batch_sequence = 0;
    uint32_t position_in_mux_batch = 0;
    uint32_t inference_batch_sequence = 0;
    uint32_t position_in_inference_batch = 0;

    uint32_t source_id = 0;
    uint32_t pad_index = 0;
    uint64_t nvds_frame_num = 0;

    int64_t pts_ns = -1;
    int64_t duration_ns = -1;
    bool pts_derived = false;

    uint32_t coded_width = 0;
    uint32_t coded_height = 0;
    uint32_t display_width = 0;
    uint32_t display_height = 0;
    int32_t rotation_degrees = 0;

    bool operator==(const FrameIdentity& other) const noexcept {
        return presentation_index == other.presentation_index &&
               decoded_sequence == other.decoded_sequence &&
               sampled_sequence == other.sampled_sequence &&
               mux_batch_sequence == other.mux_batch_sequence &&
               position_in_mux_batch == other.position_in_mux_batch &&
               inference_batch_sequence == other.inference_batch_sequence &&
               position_in_inference_batch == other.position_in_inference_batch &&
               source_id == other.source_id &&
               pad_index == other.pad_index &&
               nvds_frame_num == other.nvds_frame_num &&
               pts_ns == other.pts_ns &&
               duration_ns == other.duration_ns &&
               pts_derived == other.pts_derived &&
               coded_width == other.coded_width &&
               coded_height == other.coded_height &&
               display_width == other.display_width &&
               display_height == other.display_height &&
               rotation_degrees == other.rotation_degrees;
    }
    bool operator!=(const FrameIdentity& other) const noexcept {
        return !(*this == other);
    }
};

// Frame identity plus the device view and retained owner needed for
// batched GPU inference. Ownership semantics: |owner| must outlive any
// kernel launch that reads |device_view|.
struct FrameEnvelope : public FrameIdentity {
    DeviceImageView device_view;
    std::shared_ptr<RetainedBufferHandle> owner;
};

}  // namespace mergenvision::video
