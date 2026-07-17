#include "model_profile.h"

#include <pybind11/pybind11.h>
#include <sstream>

namespace py = pybind11;

namespace mergenvision {

namespace {

template <typename T>
T get_scalar(const py::dict& d, const char* key) {
    if (!d.contains(key)) {
        throw std::runtime_error(std::string("model profile missing key: ") + key);
    }
    try {
        return d[key].cast<T>();
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("model profile key cast failed for ") + key + ": " + e.what());
    }
}

const py::dict get_dict(const py::dict& d, const char* key) {
    if (!d.contains(key)) {
        throw std::runtime_error(std::string("model profile missing key: ") + key);
    }
    try {
        return d[key].cast<py::dict>();
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("model profile key not a dict: ") + key + ": " + e.what());
    }
}

const py::list get_list(const py::dict& d, const char* key) {
    if (!d.contains(key)) {
        throw std::runtime_error(std::string("model profile missing key: ") + key);
    }
    try {
        return d[key].cast<py::list>();
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("model profile key not a list: ") + key + ": " + e.what());
    }
}

void validate_4d_shape(const py::list& shape, const char* label) {
    if (shape.size() < 4) {
        throw std::runtime_error(std::string(label) + " shape must have 4 dimensions");
    }
}

} // namespace

ModelProfile ModelProfile::from_py_dict(const py::dict& d) {
    ModelProfile p;

    p.model_version = get_scalar<std::string>(d, "model_version");
    p.preprocess_version = get_scalar<std::string>(d, "preprocess_version");

    py::dict det = get_dict(d, "detector");
    p.detector_input_name = get_scalar<std::string>(det, "input_tensor_name");
    p.detector_conf_threshold = get_scalar<float>(det, "confidence_threshold");
    p.detector_nms_threshold = get_scalar<float>(det, "nms_threshold");
    p.detector_max_candidates = get_scalar<int>(det, "max_candidates");
    p.detector_anchor_strides = get_scalar<std::vector<int>>(det, "anchor_strides");
    p.detector_anchor_sizes = get_scalar<std::vector<std::vector<int>>>(det, "anchor_sizes");
    p.detector_anchor_ratios = get_scalar<std::vector<float>>(det, "anchor_ratios");
    p.detector_variances = get_scalar<std::vector<float>>(det, "variances");

    py::dict det_outputs = get_dict(det, "output_tensors");
    p.detector_loc_name = get_scalar<std::string>(det_outputs, "location");
    p.detector_conf_name = get_scalar<std::string>(det_outputs, "confidence");
    p.detector_landms_name = get_scalar<std::string>(det_outputs, "landmarks");

    py::list det_shape = get_list(det, "input_shape");
    validate_4d_shape(det_shape, "detector input");
    p.detector_input_size = det_shape[2].cast<int>();

    py::dict det_profile = get_dict(det, "dynamic_profile");
    py::list det_max = get_list(det_profile, "max");
    validate_4d_shape(det_max, "detector dynamic_profile.max");
    if (det_max[0].cast<int>() != 8 || det_max[1].cast<int>() != 3 ||
        det_max[2].cast<int>() != p.detector_input_size ||
        det_max[3].cast<int>() != p.detector_input_size) {
        throw std::runtime_error("detector dynamic_profile.max does not match [8,3,H,W] contract");
    }

    py::dict rec = get_dict(d, "recognizer");
    p.recognizer_input_name = get_scalar<std::string>(rec, "input_tensor_name");
    p.recognizer_embedding_dim = get_scalar<int>(rec, "embedding_dim");

    py::dict rec_outputs = get_dict(rec, "output_tensors");
    p.recognizer_output_name = get_scalar<std::string>(rec_outputs, "embedding");

    py::list rec_shape = get_list(rec, "input_shape");
    validate_4d_shape(rec_shape, "recognizer input");
    p.recognizer_input_h = rec_shape[2].cast<int>();
    p.recognizer_input_w = rec_shape[3].cast<int>();

    py::dict rec_profile = get_dict(rec, "dynamic_profile");
    py::list rec_max = get_list(rec_profile, "max");
    validate_4d_shape(rec_max, "recognizer dynamic_profile.max");
    if (rec_max[0].cast<int>() != 32 || rec_max[1].cast<int>() != 3 ||
        rec_max[2].cast<int>() != p.recognizer_input_h ||
        rec_max[3].cast<int>() != p.recognizer_input_w) {
        throw std::runtime_error("recognizer dynamic_profile.max does not match [32,3,H,W] contract");
    }

    py::dict align = get_dict(d, "alignment");
    py::list crop_size = get_list(align, "crop_size");
    if (crop_size.size() != 2) {
        throw std::runtime_error("alignment crop_size must be [height, width]");
    }
    p.alignment_crop_h = crop_size[0].cast<int>();
    p.alignment_crop_w = crop_size[1].cast<int>();
    if (p.alignment_crop_h != p.alignment_crop_w) {
        throw std::runtime_error("alignment crop_size must be square [S, S]");
    }
    p.alignment_crop_size = p.alignment_crop_h;

    py::list tmpl = get_list(align, "template");
    if (tmpl.size() != 5) {
        throw std::runtime_error("alignment template must have 5 landmarks");
    }
    p.alignment_template.clear();
    for (size_t i = 0; i < tmpl.size(); ++i) {
        py::list pt = tmpl[i].cast<py::list>();
        if (pt.size() != 2) {
            throw std::runtime_error("alignment template point must be [x, y]");
        }
        p.alignment_template.push_back({pt[0].cast<float>(), pt[1].cast<float>()});
    }

    return p;
}

} // namespace mergenvision
