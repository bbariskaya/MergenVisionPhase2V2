#pragma once

#include "mv/video/frame_identity.hpp"

#include <atomic>
#include <utility>

namespace mergenvision::video {

// Test-only retained handle that lets unit tests prove lifetime rules
// without needing real GStreamer buffers.
class FakeRetainedBufferHandle : public RetainedBufferHandle {
public:
    explicit FakeRetainedBufferHandle(int id) : id_(id) {}

    ~FakeRetainedBufferHandle() override {
        release();
    }

    int id() const { return id_; }

    void release() override {
        released_.store(true);
    }

    bool is_released() const override {
        return released_.load();
    }

private:
    int id_ = 0;
    std::atomic<bool> released_{false};
};

// Helper to create a shared retained handle.
inline std::shared_ptr<RetainedBufferHandle> make_fake_retained_handle(int id) {
    return std::make_shared<FakeRetainedBufferHandle>(id);
}

}  // namespace mergenvision::video
