#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace mergenvision {

/* Runtime contract parsed from the model profile JSON.
 *
 * The Python control plane owns JSON parsing/Pydantic validation and passes a
 * plain dict to the native extension. This struct extracts only the fields the
 * native pipeline needs, keeping the C++ side dependency-free.
 */
struct ModelProfile {
    std::string model_version;
    std::string preprocess_version;

    std::string detector_input_name;
    std::string detector_loc_name;
    std::string detector_conf_name;
    std::string detector_landms_name;
    int detector_input_size = 640;
    float detector_conf_threshold = 0.5f;
    float detector_nms_threshold = 0.4f;
    int detector_max_candidates = 300;
    std::vector<int> detector_anchor_strides = {8, 16, 32};
    std::vector<std::vector<int>> detector_anchor_sizes = {{16, 32}, {64, 128}, {256, 512}};
    std::vector<float> detector_anchor_ratios = {1.0f};
    std::vector<float> detector_variances = {0.1f, 0.2f};

    std::string recognizer_input_name;
    std::string recognizer_output_name;
    int recognizer_input_h = 112;
    int recognizer_input_w = 112;
    int recognizer_embedding_dim = 512;

    std::vector<std::vector<float>> alignment_template = {
        {38.2946f, 51.6963f},
        {73.5318f, 51.5014f},
        {56.0252f, 71.7366f},
        {41.5493f, 92.3655f},
        {70.7299f, 92.2041f},
    };
    int alignment_crop_h = 112;
    int alignment_crop_w = 112;
    int alignment_crop_size = 112;  // legacy alias, kept equal to alignment_crop_h

    static ModelProfile from_py_dict(const py::dict& d);
};

} // namespace mergenvision
