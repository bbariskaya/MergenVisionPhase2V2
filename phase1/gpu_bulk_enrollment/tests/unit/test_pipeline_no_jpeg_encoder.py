"""Regression: the extraction pipeline must not construct a GPU JPEG encoder.

MergenVisionDemo persists the original source bytes to MinIO as octet-stream
instead of re-encoding aligned chips.  Removing the encoder dependency fixes
fail-closed startup failures on runtimes where the only available JPEG encode
backend is CPU-only or unsupported by the driver.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("cuda.bindings")

from cuda.bindings import runtime as cuda_runtime  # noqa: E402
from mv_phase1_bulk import pipeline as pipeline_module  # noqa: E402
from mv_phase1_bulk.pipeline import GpuFacePipeline  # noqa: E402
from nvidia import nvimgcodec  # noqa: E402


def _fake_profile() -> dict[str, Any]:
    return {
        "model_version": "mv1",
        "preprocess_version": "pv1",
        "detector": {
            "input_shape": [1, 3, 640, 640],
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "max_candidates": 2000,
        },
        "recognizer": {"embedding_dim": 512},
        "engine_manifest": {
            "retinaface_r50_dynamic": {"engine_path": "does_not_matter"},
            "glintr100": {"engine_path": "does_not_matter"},
        },
    }


class _EverythingNoOp:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


async def test_pipeline_does_not_construct_jpeg_encoder(monkeypatch: Any) -> None:
    encoder_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class _RecordingEncoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            encoder_calls.append((args, kwargs))

    monkeypatch.setattr(nvimgcodec, "Encoder", _RecordingEncoder)
    monkeypatch.setattr(pipeline_module, "JpegGpuDecoder", _EverythingNoOp)
    monkeypatch.setattr(pipeline_module, "RetinaFacePreprocessor", _EverythingNoOp)
    monkeypatch.setattr(pipeline_module, "TrtDeviceEngine", _EverythingNoOp)
    monkeypatch.setattr(pipeline_module, "RetinaFacePostprocess", _EverythingNoOp)
    monkeypatch.setattr(pipeline_module, "GpuFaceAligner", _EverythingNoOp)
    monkeypatch.setattr(pipeline_module, "GpuRecognizer", _EverythingNoOp)
    monkeypatch.setattr(pipeline_module, "BufferArena", _EverythingNoOp)
    monkeypatch.setattr(cuda_runtime, "cudaSetDevice", lambda _dev: 0)
    monkeypatch.setattr(cuda_runtime, "cudaStreamCreate", lambda: (0, 0))
    monkeypatch.setattr(cuda_runtime, "cudaStreamDestroy", lambda _stream: 0)

    old_streams = dict(pipeline_module._SHARED_STREAMS)
    pipeline_module._SHARED_STREAMS.clear()
    try:
        pipeline = GpuFacePipeline(
            model_profile=_fake_profile(),
            device_id=0,
        )
        assert not encoder_calls, (
            f"nvimgcodec.Encoder was constructed {len(encoder_calls)} time(s); pipeline must not depend on JPEG encode"
        )
        pipeline.close()
    finally:
        pipeline_module._SHARED_STREAMS.clear()
        pipeline_module._SHARED_STREAMS.update(old_streams)
