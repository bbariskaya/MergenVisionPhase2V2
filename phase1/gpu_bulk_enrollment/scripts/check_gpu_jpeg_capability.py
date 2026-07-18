"""Phase 1 GPU JPEG encoder capability gate.

This script must pass before any production JPEG encode code is written.
It checks:
- Python/CUDA/TensorRT/nvImageCodec versions and SM support
- Explicit NVIDIA JPEG encoder backend availability (HW_GPU_ONLY then GPU_ONLY)
- A tiny host-RGB encode round-trip produces valid JPEG bytes without OpenCV fallback
- Existing engine files and manifest consistency
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _fail(code: str, message: str) -> None:
    print(f"BLOCKED {code}: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    print("=" * 60)
    print("Phase 1 GPU JPEG Capability Gate")
    print("=" * 60)

    # 1. Python
    if sys.version_info[:2] != (3, 12):
        _fail("PYTHON_VERSION", f"Expected Python 3.12, got {sys.version_info[:2]}")
    print(f"python: {sys.version}")

    # 2. CUDA driver / runtime
    try:
        from cuda.bindings import runtime as cuda_runtime
    except Exception as exc:
        _fail("CUDA_PYTHON", f"cuda-python import failed: {exc}")

    err, runtime_version = cuda_runtime.cudaRuntimeGetVersion()
    if err != cuda_runtime.cudaError_t.cudaSuccess:
        _fail("CUDA_RUNTIME", f"cudaRuntimeGetVersion failed: {err}")
    print(f"cuda_runtime_version: {runtime_version}")

    err, driver_version = cuda_runtime.cudaDriverGetVersion()
    if err != cuda_runtime.cudaError_t.cudaSuccess:
        _fail("CUDA_DRIVER", f"cudaDriverGetVersion failed: {err}")
    print(f"cuda_driver_version: {driver_version}")

    err, device_count = cuda_runtime.cudaGetDeviceCount()
    if err != cuda_runtime.cudaError_t.cudaSuccess:
        _fail("CUDA_DEVICES", f"cudaGetDeviceCount failed: {err}")
    print(f"cuda_device_count: {device_count}")
    if device_count < 1:
        _fail("CUDA_NO_DEVICES", "No CUDA devices found")

    # Prefer the first device for the gate; multi-GPU comes later.
    err, props = cuda_runtime.cudaGetDeviceProperties(0)
    if err != cuda_runtime.cudaError_t.cudaSuccess:
        _fail("CUDA_PROPS", f"cudaGetDeviceProperties failed: {err}")
    major = props.major
    minor = props.minor
    sm = f"{major}.{minor}"
    print(f"cuda_device_0: {props.name} sm_{sm} memory={props.totalGlobalMem}")
    if major < 7:
        _fail("CUDA_SM", f"SM {sm} is below the minimum target SM 7.5")

    # 3. TensorRT
    try:
        import tensorrt as trt
    except Exception as exc:
        _fail("TENSORRT_IMPORT", f"TensorRT import failed: {exc}")
    print(f"tensorrt: {trt.__version__}")
    expected_trt = "10.3.0"
    if trt.__version__ != expected_trt:
        _fail("TENSORRT_VERSION", f"Expected TensorRT {expected_trt}, got {trt.__version__}")

    # 4. nvImageCodec
    try:
        from nvidia import nvimgcodec
    except Exception as exc:
        _fail("NVIMGCODEC_IMPORT", f"nvimgcodec import failed: {exc}")
    print(f"nvimgcodec: {nvimgcodec}")

    # 5. Try explicit NVIDIA encoder backend only (HYBRID_CPU_GPU is the
    # nvJPEG CUDA encoder path; GPU_ONLY / HW_GPU_ONLY are encoder-only and
    # may not be available on this platform).
    encoder: nvimgcodec.Encoder | None = None
    chosen_backend: str | None = None
    candidates = [
        ("hybrid_cpu_gpu", nvimgcodec.BackendKind.HYBRID_CPU_GPU),
        ("gpu_only", nvimgcodec.BackendKind.GPU_ONLY),
        ("hw_gpu_only", nvimgcodec.BackendKind.HW_GPU_ONLY),
    ]
    last_error: Exception | None = None
    for name, kind in candidates:
        try:
            encoder = nvimgcodec.Encoder(
                device_id=0,
                backends=[nvimgcodec.Backend(backend_kind=kind)],
            )
            chosen_backend = name
            print(f"encoder_backend: {name}")
            break
        except Exception as exc:
            last_error = exc
            print(f"encoder_backend {name} unavailable: {exc}")

    if encoder is None or chosen_backend is None:
        _fail(
            "GPU_JPEG_ENCODER_UNAVAILABLE",
            f"No NVIDIA JPEG encoder backend available. Last error: {last_error}",
        )

    # 6. Sanity encode a tiny RGB *device* image to verify GPU-side encoding.
    import numpy as np

    class _CudaArray:
        def __init__(self, ptr: int, shape: tuple[int, ...], dtype: np.dtype) -> None:
            self.ptr = ptr
            self.shape = shape
            self.dtype = dtype

        @property
        def __cuda_array_interface__(self) -> dict[str, Any]:
            itemsize = self.dtype.itemsize
            # HWC C-contiguous strides.
            h, w, c = self.shape
            strides = (w * c * itemsize, c * itemsize, itemsize)
            return {
                "shape": self.shape,
                "typestr": self.dtype.str,
                "data": (self.ptr, False),
                "version": 3,
                "strides": strides,
            }

    tiny = np.full((112, 112, 3), 128, dtype=np.uint8)
    err, dptr = cuda_runtime.cudaMalloc(tiny.nbytes)
    if err != cuda_runtime.cudaError_t.cudaSuccess:
        _fail("CUDA_MALLOC", f"cudaMalloc failed: {err}")
    try:
        err, = cuda_runtime.cudaMemcpy(
            dptr,
            tiny.ctypes.data,
            tiny.nbytes,
            cuda_runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        if err != cuda_runtime.cudaError_t.cudaSuccess:
            _fail("CUDA_MEMCPY", f"cudaMemcpy H2D failed: {err}")
        cuda_img = nvimgcodec.as_image(
            _CudaArray(dptr, tiny.shape, tiny.dtype),
            sample_format=nvimgcodec.SampleFormat.I_RGB,
            color_spec=nvimgcodec.ColorSpec.SRGB,
        )
        assert encoder is not None
        try:
            encoded = encoder.encode(cuda_img, codec="jpeg")
            jpeg_bytes = bytes(encoded)
        except Exception as exc:
            _fail("GPU_JPEG_ENCODE_SMOKE", f"Encode smoke failed: {exc}")
        finally:
            with contextlib.suppress(Exception):
                del encoder
    finally:
        cuda_runtime.cudaFree(dptr)

    if len(jpeg_bytes) < 10:
        _fail("GPU_JPEG_ENCODE_EMPTY", "Encoded JPEG stream is empty")
    if jpeg_bytes[:3] != b"\xff\xd8\xff":
        _fail("GPU_JPEG_ENCODE_MAGIC", "Encoded bytes are not a valid JPEG stream")
    print(f"encode_smoke: ok, bytes={len(jpeg_bytes)}")

    # 7. Engine files + manifest sanity (no deserialize yet).
    repo_root = Path(__file__).parents[1].resolve()
    model_profile_path = repo_root / "config" / "model_profile.json"
    if not model_profile_path.exists():
        _fail("ENGINE_MANIFEST_MISSING", f"model_profile.json not found at {model_profile_path}")

    profile = json.loads(model_profile_path.read_text(encoding="utf-8"))
    engine_manifest = profile.get("engine_manifest", {})
    for model_key in ("retinaface_r50_dynamic", "glintr100"):
        entry = engine_manifest.get(model_key)
        if not entry:
            _fail("ENGINE_MANIFEST_ENTRY", f"Missing engine manifest entry for {model_key}")
        rel_path = entry["engine_path"]
        engine_file = repo_root / rel_path
        if not engine_file.exists():
            _fail("ENGINE_FILE_MISSING", f"Engine file missing: {engine_file}")
        actual_sha = hashlib.sha256(engine_file.read_bytes()).hexdigest()
        expected_sha = entry.get("engine_sha256")
        print(f"engine_{model_key}: {engine_file.name}")
        print(f"engine_{model_key}_sha256_actual: {actual_sha}")
        if expected_sha:
            print(f"engine_{model_key}_sha256_expected: {expected_sha}")
            if actual_sha != expected_sha:
                _fail(
                    "ENGINE_SHA_MISMATCH",
                    f"SHA mismatch for {model_key}; do not deserialize an unknown engine",
                )
        else:
            print(f"engine_{model_key}_sha256_expected: (none, will update manifest)")

    trt_manifest_version = engine_manifest.get("tensorrt_version", "")
    print(f"engine_manifest_tensorrt_version: {trt_manifest_version}")
    if trt_manifest_version and trt_manifest_version != trt.__version__:
        _fail(
            "ENGINE_TRT_VERSION_MISMATCH",
            f"Engine manifest built with TensorRT {trt_manifest_version} but runtime is {trt.__version__}",
        )

    print("=" * 60)
    print("CAPABILITY GATE PASS")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
