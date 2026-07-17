cmake_minimum_required(VERSION 3.20)
project(image_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

find_library(NVINFER_LIBRARY nvinfer
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/tensorrt/lib)
find_path(NVINFER_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu /usr/local/tensorrt/include)
if(NOT NVINFER_LIBRARY OR NOT NVINFER_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT (nvinfer) not found")
endif()

find_library(NVJPEG_LIBRARY nvjpeg PATHS ${CUDAToolkit_LIBRARY_DIR})
if(NOT NVJPEG_LIBRARY)
    message(FATAL_ERROR "nvjpeg not found")
endif()

find_library(WEBP_LIBRARY webp)
if(NOT WEBP_LIBRARY)
    message(FATAL_ERROR "libwebp not found; install libwebp-dev")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NVINFER_INCLUDE_DIR}
)

set(KERNEL_SOURCES
    src/kernels/retinaface_decode.cu
    src/kernels/argsort.cu
    src/kernels/nms.cu
    src/kernels/scale_clip_compact_xy.cu
    src/kernels/l2_normalize.cu
    src/kernels/similarity_transform.cu
    src/kernels/warp_align.cu
    src/kernels/preprocess_bgr.cu
)

# Build CUDA kernels as a static library first so CMake runs the device-link
# step and resolves the per-object fatbinData symbols.
add_library(image_runtime_cuda STATIC ${KERNEL_SOURCES})
set_target_properties(image_runtime_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(image_runtime_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
target_link_libraries(image_runtime_cuda PRIVATE CUDA::cudart CUDA::cuda_driver)

pybind11_add_module(image_runtime
    src/bindings.cpp
    src/pipeline.cpp
    src/jpeg_decoder.cpp
    src/webp_encoder.cpp
    src/retinaface_engine.cpp
    src/retinaface_postproc.cpp
    src/glintr100_engine.cpp
)

set_target_properties(image_runtime PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(image_runtime PRIVATE
    image_runtime_cuda
    CUDA::cudart
    CUDA::cuda_driver
    ${NVINFER_LIBRARY}
    ${NVJPEG_LIBRARY}
    ${WEBP_LIBRARY}
)

install(TARGETS image_runtime
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .)
