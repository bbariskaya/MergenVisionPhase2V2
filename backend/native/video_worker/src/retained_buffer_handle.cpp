#include "mv/video/retained_buffer_handle.hpp"

namespace mergenvision::video {

GstBufferRetainer::GstBufferRetainer(GstBuffer* buffer) : buffer_(buffer) {
    if (buffer_) {
        gst_buffer_ref(buffer_);
    }
}

GstBufferRetainer::~GstBufferRetainer() {
    release();
}

void GstBufferRetainer::release() {
    bool expected = false;
    if (released_.compare_exchange_strong(expected, true)) {
        if (buffer_) {
            gst_buffer_unref(buffer_);
            buffer_ = nullptr;
        }
    }
}

bool GstBufferRetainer::is_released() const {
    return released_.load();
}

} // namespace mergenvision::video
