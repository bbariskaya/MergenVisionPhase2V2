"""Inline TensorRT engine builder from ONNX for Phase 1 bulk enrollment."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import tensorrt as trt

logger = logging.getLogger(__name__)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _validate_onnx(profile: dict[str, Any], repo_root: Path) -> None:
    for key, model in profile["models"].items():
        onnx_path = Path(model["onnx_path"])
        if not onnx_path.is_absolute():
            onnx_path = repo_root / onnx_path
        expected = model["onnx_sha256"]
        actual = _sha256(onnx_path)
        if actual != expected:
            raise RuntimeError(
                f"ONNX SHA256 mismatch for {key}: expected {expected}, got {actual}"
            )
        logger.info("%s ONNX SHA256 OK: %s", key, actual)


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    input_name: str,
    input_shape_min: list[int],
    input_shape_opt: list[int],
    input_shape_max: list[int],
    fp16: bool = True,
    workspace_mb: int = 4096,
) -> None:
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    builder = trt.Builder(TRT_LOGGER)
    try:
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
    except AttributeError:
        # TensorRT 11 defaults to explicit batch.
        network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024
    )
    if fp16 and hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
    elif fp16:
        logger.warning("TensorRT %s does not expose BuilderFlag.FP16; building fp32", trt.__version__)

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, input_shape_min, input_shape_opt, input_shape_max)
    config.add_optimization_profile(profile)

    logger.info(
        "Building engine for %s -> %s (min=%s opt=%s max=%s fp16=%s)",
        onnx_path,
        engine_path,
        input_shape_min,
        input_shape_opt,
        input_shape_max,
        fp16,
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Engine build failed for {onnx_path}")

    engine_bytes = bytes(serialized)
    engine_path.write_bytes(engine_bytes)
    logger.info(
        "Engine built: %s (%d bytes, sha256=%s)",
        engine_path,
        len(engine_bytes),
        hashlib.sha256(engine_bytes).hexdigest(),
    )


def build_engines(profile_path: Path, workspace_mb: int = 4096) -> dict[str, Any]:
    profile_path = profile_path.resolve()
    repo_root = profile_path.parents[3]

    with profile_path.open("r", encoding="utf-8") as f:
        profile = json.load(f)

    _validate_onnx(profile, repo_root)

    # Detector
    det = profile["detector"]
    det_onnx = repo_root / profile["models"]["retinaface_r50_dynamic"]["onnx_path"]
    det_engine = repo_root / profile["engine_manifest"]["retinaface_r50_dynamic"][
        "engine_path"
    ]
    det_profile = det["dynamic_profile"]
    _build_engine(
        det_onnx,
        det_engine,
        det["input_tensor_name"],
        det_profile["min"],
        det_profile["opt"],
        det_profile["max"],
        fp16=profile["engine_manifest"]["retinaface_r50_dynamic"].get("precision")
        == "fp16",
        workspace_mb=workspace_mb,
    )
    profile["engine_manifest"]["retinaface_r50_dynamic"]["engine_sha256"] = _sha256(
        det_engine
    )

    # Recognizer
    rec = profile["recognizer"]
    rec_onnx = repo_root / profile["models"]["glintr100"]["onnx_path"]
    rec_engine = repo_root / profile["engine_manifest"]["glintr100"]["engine_path"]
    rec_profile = rec["dynamic_profile"]
    _build_engine(
        rec_onnx,
        rec_engine,
        rec["input_tensor_name"],
        rec_profile["min"],
        rec_profile["opt"],
        rec_profile["max"],
        fp16=profile["engine_manifest"]["glintr100"].get("precision") == "fp16",
        workspace_mb=workspace_mb,
    )
    profile["engine_manifest"]["glintr100"]["engine_sha256"] = _sha256(rec_engine)

    # Update runtime metadata
    profile["engine_manifest"]["tensorrt_version"] = trt.__version__
    profile["engine_manifest"]["build_timestamp"] = (
        datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    with profile_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    logger.info("Updated model profile: %s", profile_path)
    return profile
