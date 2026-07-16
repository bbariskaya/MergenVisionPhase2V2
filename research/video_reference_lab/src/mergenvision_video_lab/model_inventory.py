"""Local model discovery, hashing, ONNX inspection, and provider selection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort

from mergenvision_video_lab.contracts import OnnxModelContract
from mergenvision_video_lab.errors import ConfigError, ModelArtifactError
from mergenvision_video_lab.hashing import sha256_file


DEFAULT_DETECTOR_BASENAMES = ["retinaface_r50_dynamic.onnx"]
DEFAULT_RECOGNIZER_BASENAMES = ["glintr100.onnx"]


def _repo_root() -> Path:
    """Best-effort repository root resolution from this file's location."""
    # File is at research/video_reference_lab/src/mergenvision_video_lab/...
    return Path(__file__).resolve().parents[4]


def resolve_model_root(configured_root: str | None) -> Path:
    """Resolve model root relative to repository root if not absolute."""
    if configured_root is None:
        return _repo_root() / "backend" / "artifacts" / "models"
    path = Path(configured_root)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _find_artifact(
    directory: Path,
    candidates: list[str],
    role: str,
) -> Path:
    """Find a model artifact in the directory."""
    discovered = list(directory.iterdir()) if directory.exists() else []
    for basename in candidates:
        candidate = directory / basename
        if candidate.exists():
            return candidate
    raise ModelArtifactError(
        f"{role} artifact not found",
        {
            "directory": str(directory),
            "expected_basenames": candidates,
            "discovered_files": [p.name for p in discovered],
        },
    )


def _inspect_onnx(path: Path) -> dict[str, Any]:
    """Inspect ONNX graph and return a serializable contract."""
    try:
        model = onnx.load(str(path))
    except Exception as exc:
        raise ModelArtifactError(f"cannot load ONNX model {path.name}: {exc}") from exc

    inputs = []
    for inp in model.graph.input:
        tensor_type = inp.type.tensor_type
        shape = [
            dim.dim_value if dim.dim_value > 0 else dim.dim_param or "dynamic"
            for dim in tensor_type.shape.dim
        ]
        inputs.append(
            {
                "name": inp.name,
                "shape": shape,
                "dtype": onnx.helper.tensor_dtype_to_np_dtype(
                    tensor_type.elem_type
                ).name,
            }
        )

    outputs = []
    for out in model.graph.output:
        tensor_type = out.type.tensor_type
        shape = [
            dim.dim_value if dim.dim_value > 0 else dim.dim_param or "dynamic"
            for dim in tensor_type.shape.dim
        ]
        outputs.append(
            {
                "name": out.name,
                "shape": shape,
                "dtype": onnx.helper.tensor_dtype_to_np_dtype(
                    tensor_type.elem_type
                ).name,
            }
        )

    return {
        "inputs": inputs,
        "outputs": outputs,
        "opset": model.opset_import[0].version if model.opset_import else None,
        "producer": model.producer_name or None,
    }


def build_model_contract(path: Path) -> OnnxModelContract:
    """Build an OnnxModelContract from a local ONNX file."""
    inspection = _inspect_onnx(path)
    return OnnxModelContract(
        basename=path.name,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        inputs=inspection["inputs"],
        outputs=inspection["outputs"],
        opset=inspection["opset"],
        producer=inspection["producer"],
    )


class ModelInventory:
    """Discovered local detector and recognizer artifacts."""

    def __init__(
        self,
        model_root: str | None,
        model_pack: str | None,
        detector_basenames: list[str] | None = None,
        recognizer_basenames: list[str] | None = None,
    ) -> None:
        self.root = resolve_model_root(model_root)
        if model_pack is not None:
            self.root = self.root / model_pack
        self.detector_path = _find_artifact(
            self.root,
            detector_basenames or DEFAULT_DETECTOR_BASENAMES,
            "detector",
        )
        self.recognizer_path = _find_artifact(
            self.root,
            recognizer_basenames or DEFAULT_RECOGNIZER_BASENAMES,
            "recognizer",
        )
        self.detector_contract = build_model_contract(self.detector_path)
        self.recognizer_contract = build_model_contract(self.recognizer_path)


def available_providers() -> list[str]:
    """Return ONNX Runtime available execution providers."""
    try:
        return ort.get_available_providers()
    except Exception as exc:
        raise ConfigError(f"cannot query ONNX Runtime providers: {exc}") from exc


def select_providers(
    requested: str,
    available: list[str],
    allow_cpu_fallback: bool,
) -> list[str]:
    """Select actual ONNX Runtime providers.

    Never silently fall back to CPU. CUDA requested but unavailable fails
    unless ``allow_cpu_fallback`` is True.
    """
    if requested == "cuda":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if allow_cpu_fallback:
            return ["CPUExecutionProvider"]
        raise ModelArtifactError(
            "CUDA requested but not available and CPU fallback not allowed",
            {
                "requested": requested,
                "available": available,
                "allow_cpu_fallback": allow_cpu_fallback,
            },
        )
    if requested == "cpu":
        if "CPUExecutionProvider" not in available:
            raise ModelArtifactError(
                "CPUExecutionProvider not available",
                {"requested": requested, "available": available},
            )
        return ["CPUExecutionProvider"]
    raise ConfigError(f"unsupported provider: {requested}")
