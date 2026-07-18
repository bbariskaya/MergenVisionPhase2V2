"""Verify the aligned-chip JPEG encode path stays on GPU until compressed bytes."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

pytest.importorskip("cuda.bindings")

from cuda.bindings import runtime as cuda_runtime
from mv_phase1_bulk._gpu_ops import nchw_float_to_hwc_uint8
from mv_phase1_bulk.device_tensor import check_cuda
from nvidia import nvimgcodec


def _check(err: Any, msg: str) -> None:
    check_cuda(err, msg)


def test_nchw_float_to_jpeg_roundtrip() -> None:
    """Float NCHW chip -> GPU uint8 HWC -> nvImageCodec JPEG -> valid bytes."""
    h = w = 112
    n = 1
    chip_f = np.random.RandomState(42).uniform(0, 255, size=(n, 3, h, w)).astype(np.float32)

    err, d_src = cuda_runtime.cudaMalloc(chip_f.nbytes)
    _check(err, "cudaMalloc src")
    err, d_dst = cuda_runtime.cudaMalloc(n * h * w * 3)
    _check(err, "cudaMalloc dst")

    try:
        err, = cuda_runtime.cudaMemcpy(
            d_src,
            chip_f.ctypes.data,
            chip_f.nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        _check(err, "cudaMemcpy H2D")
        cuda_runtime.cudaStreamSynchronize(0)

        nchw_float_to_hwc_uint8(d_src, d_dst, n, h, w, 0)
        cuda_runtime.cudaStreamSynchronize(0)

        class _CudaHwC:
            def __init__(self, ptr: int, shape: tuple[int, int, int]) -> None:
                self.ptr = ptr
                self.shape = shape

            @property
            def __cuda_array_interface__(self) -> dict[str, Any]:
                hh, ww, cc = self.shape
                return {
                    "shape": self.shape,
                    "typestr": "|u1",
                    "data": (self.ptr, False),
                    "version": 3,
                    "strides": (ww * cc, cc, 1),
                }

        encoder = nvimgcodec.Encoder(
            device_id=0,
            backends=[nvimgcodec.Backend(backend_kind=nvimgcodec.BackendKind.HYBRID_CPU_GPU)],
        )
        cuda_img = nvimgcodec.as_image(
            _CudaHwC(d_dst, (h, w, 3)),
            sample_format=nvimgcodec.SampleFormat.I_RGB,
            color_spec=nvimgcodec.ColorSpec.SRGB,
        )
        encoded = encoder.encode(cuda_img, "jpeg")
        jpeg_bytes = bytes(encoded)

        assert len(jpeg_bytes) > 100
        assert jpeg_bytes[:3] == b"\xff\xd8\xff"

        decoder = nvimgcodec.Decoder()
        decoded = decoder.decode(jpeg_bytes)
        arr = np.asarray(decoded.cpu())
        assert arr.shape == (h, w, 3)
        assert arr.dtype == np.uint8
    finally:
        cuda_runtime.cudaFree(d_src)
        cuda_runtime.cudaFree(d_dst)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
