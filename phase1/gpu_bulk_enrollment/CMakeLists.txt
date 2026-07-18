cmake_minimum_required(VERSION 3.18)
project(mv_phase1_bulk_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(MV_PHASE1_BULK_NATIVE_SOURCES
    native/src/bindings.cpp
    native/src/l2_normalize.cu
    native/src/argsort.cu
    native/src/nms.cu
    native/src/retinaface_decode.cu
    native/src/retinaface_pick_largest.cu
    native/src/scale_clip_compact.cu
    native/src/scale_clip_compact_xy.cu
    native/src/similarity_transform.cu
    native/src/warp_align.cu
    native/src/test_helpers.cu
)

pybind11_add_module(_mv_phase1_bulk_native ${MV_PHASE1_BULK_NATIVE_SOURCES})

target_include_directories(_mv_phase1_bulk_native PRIVATE native/include)
target_link_libraries(_mv_phase1_bulk_native PRIVATE CUDA::cudart)
set_target_properties(_mv_phase1_bulk_native PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RUNTIME_LIBRARY Shared
)

install(TARGETS _mv_phase1_bulk_native DESTINATION mv_phase1_bulk)
