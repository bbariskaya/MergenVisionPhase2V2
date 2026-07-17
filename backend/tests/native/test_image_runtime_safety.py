"""Phase 2 Milestone 0.7 — native runtime safety/contract tests.

These tests run against the real serialized engines and must be executed in an
environment where `image_runtime` can be imported (i.e. the pinned GPU/runtime
container or a matching local build).
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import pytest

image_runtime = pytest.importorskip("image_runtime")

tensorrt = pytest.importorskip("tensorrt")

REPO_ROOT = pathlib.Path(os.environ.get("MERGENVISION_REPO_ROOT", pathlib.Path(__file__).parents[3]))
PROFILE_PATH = REPO_ROOT / "backend" / "config" / "model_profiles" / "retinaface_r50_glintr100_v1.example.json"
DETECTOR_ENGINE = REPO_ROOT / "backend" / "artifacts" / "engines" / "retinaface_r50_dynamic.bs1.opt8.max64.fp16.trt1016.engine"
RECOGNIZER_ENGINE = REPO_ROOT / "backend" / "artifacts" / "engines" / "glintr100.bs1.opt8.max64.fp16.trt1016.engine"


def _load_profile() -> dict[str, Any]:
    with PROFILE_PATH.open() as fh:
        return json.load(fh)


def _engine_io_names(engine_path: pathlib.Path) -> tuple[list[str], list[str], list[str]]:
    logger = tensorrt.Logger(tensorrt.Logger.WARNING)
    runtime = tensorrt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    all_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inputs = [n for n in all_names if engine.get_tensor_mode(n) == tensorrt.TensorIOMode.INPUT]
    outputs = [n for n in all_names if engine.get_tensor_mode(n) == tensorrt.TensorIOMode.OUTPUT]
    return all_names, inputs, outputs


def test_image_runtime_constructor_rejects_broken_slots() -> None:
    """A slot whose engine(s) fail to load must not be silently added to the pool."""
    profile = _load_profile()
    with pytest.raises(RuntimeError):
        image_runtime.ImageRuntime(
            profile,
            "/nonexistent/detector.engine",
            "/nonexistent/recognizer.engine",
            0,
            1,
        )


def test_detector_profile_tensor_names_match_engine() -> None:
    profile = _load_profile()
    _, inputs, outputs = _engine_io_names(DETECTOR_ENGINE)

    assert inputs == [profile["detector"]["input_tensor_name"]], (
        f"detector input name mismatch: engine={inputs}, profile={profile['detector']['input_tensor_name']}"
    )
    expected_outputs = [
        profile["detector"]["output_tensors"]["location"],
        profile["detector"]["output_tensors"]["confidence"],
        profile["detector"]["output_tensors"]["landmarks"],
    ]
    assert outputs == expected_outputs, f"detector output names mismatch: engine={outputs}, profile={expected_outputs}"


def test_recognizer_profile_tensor_names_match_engine() -> None:
    profile = _load_profile()
    _, inputs, outputs = _engine_io_names(RECOGNIZER_ENGINE)

    assert inputs == [profile["recognizer"]["input_tensor_name"]], (
        f"recognizer input name mismatch: engine={inputs}, profile={profile['recognizer']['input_tensor_name']}"
    )
    expected_outputs = [profile["recognizer"]["output_tensors"]["embedding"]]
    assert outputs == expected_outputs, (
        f"recognizer output name mismatch: engine={outputs}, profile={expected_outputs}"
    )


def test_profile_embeddings_and_dynamic_shapes_match_contract() -> None:
    profile = _load_profile()
    assert profile["recognizer"]["embedding_dim"] == 512
    assert profile["detector"]["dynamic_profile"]["max"] == [8, 3, 640, 640]
    assert profile["recognizer"]["dynamic_profile"]["max"] == [32, 3, 112, 112]


def test_image_runtime_can_infer_no_face_image() -> None:
    if not DETECTOR_ENGINE.exists():
        pytest.skip("detector engine not present")
    rt = image_runtime.ImageRuntime(
        _load_profile(),
        str(DETECTOR_ENGINE),
        str(RECOGNIZER_ENGINE),
        0,
        1,
    )
    data = (REPO_ROOT / "frontend" / "e2e" / "fixtures" / "no-face.jpg").read_bytes()
    out = rt.infer_jpeg(data)
    assert out["detections"] == []


def test_image_runtime_can_infer_face_image() -> None:
    if not DETECTOR_ENGINE.exists():
        pytest.skip("detector engine not present")
    rt = image_runtime.ImageRuntime(
        _load_profile(),
        str(DETECTOR_ENGINE),
        str(RECOGNIZER_ENGINE),
        0,
        1,
    )
    data = (REPO_ROOT / "frontend" / "e2e" / "fixtures" / "unknown-face.jpg").read_bytes()
    out = rt.infer_jpeg(data)
    assert len(out["detections"]) >= 1
    first = out["detections"][0]
    assert len(first["embedding"]) == 512
    assert abs(sum(v * v for v in first["embedding"]) - 1.0) < 1e-3


def test_native_runtime_error_is_typed_not_abort() -> None:
    if not DETECTOR_ENGINE.exists():
        pytest.skip("detector engine not present")
    rt = image_runtime.ImageRuntime(
        _load_profile(),
        str(DETECTOR_ENGINE),
        str(RECOGNIZER_ENGINE),
        0,
        1,
    )
    # A truncated/corrupt buffer should raise a Python exception, not kill the process.
    with pytest.raises(RuntimeError):
        rt.infer_jpeg(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 8)
    # The slot must still be usable afterwards.
    data = (REPO_ROOT / "frontend" / "e2e" / "fixtures" / "unknown-face.jpg").read_bytes()
    out = rt.infer_jpeg(data)
    assert len(out["detections"]) >= 1


def test_concurrent_infer_jpeg_does_not_deadlock() -> None:
    """Two Python threads calling inference must both complete without deadlock.

    This is a structural smoke test for GIL release / slot reuse; it does not
    claim throughput scaling.
    """
    if not DETECTOR_ENGINE.exists():
        pytest.skip("detector engine not present")
    rt = image_runtime.ImageRuntime(
        _load_profile(),
        str(DETECTOR_ENGINE),
        str(RECOGNIZER_ENGINE),
        0,
        2,
    )
    data = (REPO_ROOT / "frontend" / "e2e" / "fixtures" / "unknown-face.jpg").read_bytes()

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(rt.infer_jpeg, data) for _ in range(2)]
        results = [f.result(timeout=60) for f in futures]

    for out in results:
        assert len(out["detections"]) >= 1
        assert len(out["detections"][0]["embedding"]) == 512
