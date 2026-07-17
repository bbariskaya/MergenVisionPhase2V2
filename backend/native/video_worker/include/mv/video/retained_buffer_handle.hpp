#pragma once

#include "mv/video/frame_identity.hpp"

#include <gst/gst.h>
#include <atomic>
#include <memory>

namespace mergenvision::video {

// Concrete retained ownership handle for a GstBuffer. The buffer is referenced
// on construction and released either explicitly or on destruction. Safe to
// share via std::shared_ptr.
class GstBufferRetainer : public RetainedBufferHandle {
public:
    explicit GstBufferRetainer(GstBuffer* buffer);
    ~GstBufferRetainer() override;

    void release() override;
    bool is_released() const override;

    GstBuffer* buffer() const { return buffer_; }

private:
    GstBuffer* buffer_ = nullptr;
    std::atomic<bool> released_{false};
};

// Wrap a GstBuffer in a shared retained handle.
inline std::shared_ptr<RetainedBufferHandle> retain_buffer(GstBuffer* buffer) {
    if (!buffer) return nullptr;
    return std::make_shared<GstBufferRetainer>(buffer);
}

} // namespace mergenvision::video
