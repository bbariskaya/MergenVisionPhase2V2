cmake_minimum_required(VERSION 3.20)
project(video_worker LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(DEFINED ENV{CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures" FORCE)
else()
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING "CUDA architectures")
endif()

find_package(PkgConfig REQUIRED)
find_package(CUDAToolkit REQUIRED)

pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)
pkg_check_modules(GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0)

set(DEEPSTREAM_ROOT "/opt/nvidia/deepstream/deepstream-9.0" CACHE PATH "DeepStream install root")
if(NOT EXISTS "${DEEPSTREAM_ROOT}/sources/includes/gstnvdsmeta.h")
    message(FATAL_ERROR "DeepStream headers not found at ${DEEPSTREAM_ROOT}/sources/includes")
endif()
if(NOT EXISTS "${DEEPSTREAM_ROOT}/lib/libnvdsgst_meta.so")
    message(FATAL_ERROR "DeepStream libnvdsgst_meta not found at ${DEEPSTREAM_ROOT}/lib")
endif()
if(NOT EXISTS "${DEEPSTREAM_ROOT}/lib/libnvds_meta.so")
    message(FATAL_ERROR "DeepStream libnvds_meta not found at ${DEEPSTREAM_ROOT}/lib")
endif()
if(NOT EXISTS "${DEEPSTREAM_ROOT}/lib/libnvbufsurface.so")
    message(FATAL_ERROR "DeepStream libnvbufsurface not found at ${DEEPSTREAM_ROOT}/lib")
endif()

find_library(NVINFER_LIBRARY nvinfer
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/tensorrt/lib)
find_path(NVINFER_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu /usr/local/tensorrt/include)
if(NOT NVINFER_LIBRARY OR NOT NVINFER_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT (nvinfer) not found")
endif()

find_library(PROTOBUF_LIBRARY protobuf
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_path(PROTOBUF_INCLUDE_DIR google/protobuf/message.h
    PATHS /usr/include /usr/local/include)
if(NOT PROTOBUF_LIBRARY OR NOT PROTOBUF_INCLUDE_DIR)
    message(FATAL_ERROR "Protocol Buffers C++ library not found")
endif()

find_library(WEBP_LIBRARY webp
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib)
if(NOT WEBP_LIBRARY)
    message(FATAL_ERROR "libwebp not found")
endif()

find_library(ZSTD_LIBRARY zstd
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib)
if(NOT ZSTD_LIBRARY)
    message(FATAL_ERROR "libzstd not found")
endif()

find_library(NVML_LIBRARY nvidia-ml
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64/stubs /usr/local/cuda/lib64)
if(NOT NVML_LIBRARY)
    message(FATAL_ERROR "libnvidia-ml not found")
endif()

# Pre-built CUDA kernels from the image_runtime package.
set(IMAGE_RUNTIME_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../image_runtime/build/cp312-cp312-linux_x86_64")
if(NOT EXISTS "${IMAGE_RUNTIME_BUILD_DIR}/libimage_runtime_cuda.a")
    message(FATAL_ERROR "libimage_runtime_cuda.a not found at ${IMAGE_RUNTIME_BUILD_DIR}")
endif()
add_library(image_runtime_cuda STATIC IMPORTED)
set_target_properties(image_runtime_cuda PROPERTIES
    IMPORTED_LOCATION "${IMAGE_RUNTIME_BUILD_DIR}/libimage_runtime_cuda.a"
)

# -----------------------------------------------------------------------------
# Common include directories and CUDA compile flags reused by GPU targets.
# -----------------------------------------------------------------------------
set(VIDEO_WORKER_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../image_runtime/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../image_runtime/src/kernels
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NVINFER_INCLUDE_DIR}
    "${DEEPSTREAM_ROOT}/sources/includes"
)

set(VIDEO_WORKER_LINK_DIRS
    "${DEEPSTREAM_ROOT}/lib"
)

set(VIDEO_WORKER_GPU_LIBS
    image_runtime_cuda
    CUDA::cudart
    ${NVINFER_LIBRARY}
    nvdsgst_meta
    nvds_meta
    nvbufsurface
)

# -----------------------------------------------------------------------------
# Core metadata/batching library (CPU-only)
# -----------------------------------------------------------------------------
add_library(mv_video_core STATIC
    src/batch_assembler.cpp
)

target_include_directories(mv_video_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(mv_video_core PUBLIC cxx_std_17)

target_compile_options(mv_video_core PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
)

# -----------------------------------------------------------------------------
# decode_smoke: real GStreamer/DeepStream GPU decode smoke
# -----------------------------------------------------------------------------
add_executable(decode_smoke tests/decode_smoke.cpp)

target_include_directories(decode_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_APP_INCLUDE_DIRS}
    ${GSTREAMER_VIDEO_INCLUDE_DIRS}
    "${DEEPSTREAM_ROOT}/sources/includes"
)

target_compile_options(decode_smoke PRIVATE
    ${GSTREAMER_CFLAGS_OTHER}
    ${GSTREAMER_APP_CFLAGS_OTHER}
    ${GSTREAMER_VIDEO_CFLAGS_OTHER}
    -Wall
    -Wextra
    -Wno-unused-parameter
)

target_link_directories(decode_smoke PRIVATE
    "${DEEPSTREAM_ROOT}/lib"
)

target_link_libraries(decode_smoke PRIVATE
    mv_video_core
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
    ${GSTREAMER_VIDEO_LIBRARIES}
    nvdsgst_meta
    nvds_meta
    nvbufsurface
)

# -----------------------------------------------------------------------------
# sequence_contract: deterministic CPU-only metadata contract tests
# -----------------------------------------------------------------------------
add_executable(sequence_contract tests/sequence_contract.cpp)

target_include_directories(sequence_contract PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(sequence_contract PRIVATE
    -Wall
    -Wextra
)

target_link_libraries(sequence_contract PRIVATE mv_video_core)

# -----------------------------------------------------------------------------
# video_face_pipeline: real detector/recognizer batched inference.
# -----------------------------------------------------------------------------
add_library(video_face_pipeline STATIC
    src/video_face_pipeline.cpp
    src/kernels/preprocess_detector_rgba_batch.cu
    src/kernels/preprocess_detector_nv12_batch.cu
    ../image_runtime/src/kernels/warp_align_rgba_pitch.cu
    src/kernels/warp_align_nv12_pitch.cu
    src/retained_buffer_handle.cpp
    ../image_runtime/src/retinaface_engine.cpp
    ../image_runtime/src/retinaface_postproc.cpp
    ../image_runtime/src/glintr100_engine.cpp
)

target_include_directories(video_face_pipeline PUBLIC
    ${VIDEO_WORKER_INCLUDES}
)

target_include_directories(video_face_pipeline PRIVATE
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_APP_INCLUDE_DIRS}
    ${GSTREAMER_VIDEO_INCLUDE_DIRS}
)

target_compile_options(video_face_pipeline PRIVATE
    ${GSTREAMER_CFLAGS_OTHER}
    ${GSTREAMER_APP_CFLAGS_OTHER}
    ${GSTREAMER_VIDEO_CFLAGS_OTHER}
    -Wall
    -Wextra
    -Wno-unused-parameter
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

target_link_directories(video_face_pipeline PUBLIC
    ${VIDEO_WORKER_LINK_DIRS}
)

target_link_libraries(video_face_pipeline PUBLIC
    mv_video_core
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
    ${GSTREAMER_VIDEO_LIBRARIES}
    ${VIDEO_WORKER_GPU_LIBS}
)

set_target_properties(video_face_pipeline PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF
)

# -----------------------------------------------------------------------------
# real_batching_smoke: First M5.2 gate -- decode + detector batch inference.
# -----------------------------------------------------------------------------
add_executable(real_batching_smoke tests/real_batching_smoke.cpp)

target_include_directories(real_batching_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_APP_INCLUDE_DIRS}
    ${GSTREAMER_VIDEO_INCLUDE_DIRS}
    "${DEEPSTREAM_ROOT}/sources/includes"
    ${NVINFER_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../image_runtime/src
)

target_compile_options(real_batching_smoke PRIVATE
    ${GSTREAMER_CFLAGS_OTHER}
    ${GSTREAMER_APP_CFLAGS_OTHER}
    ${GSTREAMER_VIDEO_CFLAGS_OTHER}
    -Wall
    -Wextra
    -Wno-unused-parameter
)

target_link_directories(real_batching_smoke PRIVATE
    "${DEEPSTREAM_ROOT}/lib"
)

target_link_libraries(real_batching_smoke PRIVATE
    video_face_pipeline
    mv_video_core
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
    ${GSTREAMER_VIDEO_LIBRARIES}
    ${VIDEO_WORKER_GPU_LIBS}
)

# -----------------------------------------------------------------------------
# mv_video_worker: full GPU observation + template + representative crop producer.
# -----------------------------------------------------------------------------
add_executable(mv_video_worker
    tests/real_batching_smoke.cpp
    proto/video_observation_v1.pb.cc
    proto/video_track_template_v1.pb.cc
)

target_compile_definitions(mv_video_worker PRIVATE MV_VIDEO_WORKER)

target_include_directories(mv_video_worker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/proto
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_APP_INCLUDE_DIRS}
    ${GSTREAMER_VIDEO_INCLUDE_DIRS}
    "${DEEPSTREAM_ROOT}/sources/includes"
    ${NVINFER_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../image_runtime/src
    ${PROTOBUF_INCLUDE_DIR}
)

target_compile_options(mv_video_worker PRIVATE
    ${GSTREAMER_CFLAGS_OTHER}
    ${GSTREAMER_APP_CFLAGS_OTHER}
    ${GSTREAMER_VIDEO_CFLAGS_OTHER}
    -Wall
    -Wextra
    -Wno-unused-parameter
)

target_link_directories(mv_video_worker PRIVATE
    "${DEEPSTREAM_ROOT}/lib"
)

target_link_libraries(mv_video_worker PRIVATE
    video_face_pipeline
    mv_video_core
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
    ${GSTREAMER_VIDEO_LIBRARIES}
    ${VIDEO_WORKER_GPU_LIBS}
    ${PROTOBUF_LIBRARY}
    ${WEBP_LIBRARY}
    ${ZSTD_LIBRARY}
    ${NVML_LIBRARY}
)
