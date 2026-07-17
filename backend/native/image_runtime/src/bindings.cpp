#include "pipeline.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
using namespace std::chrono_literals;

namespace mergenvision {

class ImageRuntime {
public:
    ImageRuntime(const std::string& model_profile_path,
                 const std::string& retinaface_engine_path,
                 const std::string& glintr100_engine_path,
                 int device_id,
                 int context_pool_size)
        : model_profile_path_(model_profile_path),
          device_id_(device_id),
          context_pool_size_(context_pool_size) {
        if (context_pool_size_ < 1) {
            throw std::invalid_argument("context_pool_size must be >= 1");
        }

        // Validate that the model profile file exists.
        std::ifstream check(model_profile_path);
        if (!check) {
            throw std::runtime_error("model profile not found: " + model_profile_path);
        }
        check.close();

        slots_.reserve(context_pool_size_);
        for (int i = 0; i < context_pool_size_; ++i) {
            std::string error;
            auto slot = std::make_unique<ExecutionSlot>(
                device_id_, retinaface_engine_path, glintr100_engine_path, &error);
            if (!slot || !slot->available()) {
                throw std::runtime_error("failed to initialize execution slot " +
                                         std::to_string(i) + ": " + error);
            }
            slots_.push_back(std::move(slot));
        }
    }

    py::dict infer_jpeg(const py::object& data_obj, float acquire_timeout_ms = 30000.0f) {
        py::buffer buffer = py::buffer(data_obj);
        py::buffer_info info = buffer.request();
        const void* ptr = info.ptr;
        size_t len = static_cast<size_t>(info.size);
        if (!ptr || len == 0) {
            throw std::runtime_error("EMPTY_IMAGE");
        }

        ExecutionSlot* slot = acquire_slot(acquire_timeout_ms);
        if (!slot) {
            throw std::runtime_error("GPU_OVERLOADED");
        }

        std::string error;
        InferenceResult result;
        bool ok = slot->infer_jpeg(ptr, len, &result, &error);
        release_slot(slot);
        if (!ok) {
            throw std::runtime_error(error);
        }

        py::dict out;
        out["image_width"] = result.image_width;
        out["image_height"] = result.image_height;

        py::list faces;
        for (const auto& obs : result.detections) {
            py::dict face;
            face["detection_index"] = obs.detection_index;

            py::dict bbox;
            bbox["x"] = obs.x;
            bbox["y"] = obs.y;
            bbox["width"] = obs.width;
            bbox["height"] = obs.height;
            face["bbox"] = bbox;

            py::list landmarks;
            for (size_t i = 0; i < obs.landmarks5.size(); i += 2) {
                py::list pt;
                pt.append(obs.landmarks5[i]);
                pt.append(obs.landmarks5[i + 1]);
                landmarks.append(pt);
            }
            face["landmarks5"] = landmarks;

            face["detector_confidence"] = obs.detector_confidence;

            py::list embedding;
            for (float v : obs.embedding) {
                embedding.append(v);
            }
            face["embedding"] = embedding;

            face["aligned_crop_bytes"] = py::bytes(
                reinterpret_cast<const char*>(obs.aligned_crop_bytes.data()),
                static_cast<py::size_t>(obs.aligned_crop_bytes.size()));

            faces.append(face);
        }
        out["detections"] = faces;
        return out;
    }

private:
    ExecutionSlot* acquire_slot(float timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(static_cast<int64_t>(timeout_ms));

        while (true) {
            for (auto& slot : slots_) {
                if (slot->available()) {
                    slot->acquire();
                    return slot.get();
                }
            }
            if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                return nullptr;
            }
        }
    }

    void release_slot(ExecutionSlot* slot) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            slot->release();
        }
        cv_.notify_one();
    }

    std::string model_profile_path_;
    int device_id_ = 0;
    int context_pool_size_ = 1;

    std::vector<std::unique_ptr<ExecutionSlot>> slots_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

} // namespace mergenvision

PYBIND11_MODULE(image_runtime, m) {
    m.doc() = "MergenVision native GPU image inference runtime";

    py::class_<mergenvision::ImageRuntime>(m, "ImageRuntime")
        .def(py::init<const std::string&, const std::string&, const std::string&, int, int>(),
             py::arg("model_profile_path"),
             py::arg("retinaface_engine_path"),
             py::arg("glintr100_engine_path"),
             py::arg("device_id"),
             py::arg("context_pool_size"))
        .def("infer_jpeg",
             &mergenvision::ImageRuntime::infer_jpeg,
             py::arg("encoded_bytes"),
             py::arg("acquire_timeout_ms") = 30000.0f,
             R"doc(
Run the end-to-end GPU inference pipeline on a JPEG byte buffer.

Returns a dict with keys:
  image_width, image_height, detections
Each detection contains detection_index, bbox, landmarks5,
detector_confidence, embedding (list of 512 floats), aligned_crop_bytes (WebP).
)doc");
}
