"""Explicit model acquisition and manifest building.

Only the ``models acquire`` command is allowed to trigger an official
InsightFace download. All other code paths must load already-cached models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort

from mergenvision_video_lab.config import resolve_repo_relative_path
from mergenvision_video_lab.errors import ModelArtifactError
from mergenvision_video_lab.hashing import sha256_file

BUFFALO_L_FILES = [
    "det_10g.onnx",
    "genderage.onnx",
    "1k3d68.onnx",
    "2d106det.onnx",
    "w600k_r50.onnx",
]


def _insightface_root() -> Path:
    """Return the local InsightFace model root (usually ~/.insightface)."""
    from pathlib import Path

    return Path.home() / ".insightface"


def _onnx_contract(path: Path) -> dict[str, Any]:
    """Inspect an ONNX file and return a serializable contract."""
    model = onnx.load(str(path))
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
                "dtype": onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type).name,
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
                "dtype": onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type).name,
            }
        )
    return {
        "basename": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "inputs": inputs,
        "outputs": outputs,
        "opset": model.opset_import[0].version if model.opset_import else None,
        "producer": model.producer_name or None,
    }


def _probe_session(path: Path, provider: str) -> dict[str, Any]:
    """Create a short-lived ONNX Runtime session to record actual providers."""
    available = ort.get_available_providers()
    if provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    actual = [p for p in providers if p in available]
    if not actual:
        raise ModelArtifactError(
            "no usable execution provider",
            {"requested": provider, "available": available},
        )
    sess = ort.InferenceSession(str(path), providers=actual)
    return {
        "actual_providers": sess.get_providers(),
        "input_names": [inp.name for inp in sess.get_inputs()],
        "output_names": [out.name for out in sess.get_outputs()],
    }


def acquire_model_pack(name: str, provider: str) -> dict[str, Any]:
    """Acquire an official InsightFace model pack and return a manifest.

    The actual download is delegated to ``insightface.app.FaceAnalysis``,
    which fetches the official pack into ``~/.insightface/models/``.
    """
    if name != "buffalo_l":
        raise ModelArtifactError(f"unsupported model pack: {name}")

    # Trigger the official download via insightface.utils.ensure_available.
    # This is the same helper FaceAnalysis.__init__ uses to fetch packs.
    try:
        from insightface.utils import ensure_available
    except ImportError as exc:
        raise ModelArtifactError(f"insightface not installed: {exc}") from exc

    pack_root = _insightface_root() / "models" / name
    if not pack_root.exists():
        try:
            ensure_available("models", name, root=str(_insightface_root()))
        except Exception as exc:
            raise ModelArtifactError(
                f"failed to download model pack {name}: {exc}",
            ) from exc

    if not pack_root.exists():
        raise ModelArtifactError(
            "model pack directory not found after download",
            {"pack_root": str(pack_root)},
        )

    files: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for basename in BUFFALO_L_FILES:
        path = pack_root / basename
        if not path.exists():
            missing.append(basename)
            continue
        contract = _onnx_contract(path)
        if basename in ("det_10g.onnx",):
            contract["task"] = "detection"
        elif basename == "w600k_r50.onnx":
            contract["task"] = "recognition"
        else:
            contract["task"] = "auxiliary"
        # Probe one representative session to record actual providers.
        if basename in ("det_10g.onnx", "w600k_r50.onnx"):
            probe = _probe_session(path, provider)
            contract["runtime_providers"] = probe["actual_providers"]
        files[basename] = contract

    if missing:
        raise ModelArtifactError(
            "model pack is incomplete",
            {"missing": missing, "pack_root": str(pack_root)},
        )

    # Copy to the configured lab cache so the path is repo-local and Git-ignored.
    # The lab cache root doubles as the InsightFace root, so the pack lives under
    # ``<root>/models/<pack>`` exactly as FaceAnalysis expects.
    lab_cache = resolve_repo_relative_path(
        "research/video_reference_lab/.model_cache/models/buffalo_l"
    )
    lab_cache.mkdir(parents=True, exist_ok=True)
    for basename, info in files.items():
        src = pack_root / basename
        dst = lab_cache / basename
        if not dst.exists() or sha256_file(dst) != info["sha256"]:
            dst.write_bytes(src.read_bytes())

    manifest = {
        "name": name,
        "source": "official InsightFace python-package",
        "license_notice": (
            "InsightFace code is MIT. Provided pretrained weights are for "
            "non-commercial research use only. See "
            "https://github.com/deepinsight/insightface/tree/master/python-package"
        ),
        "pack_root": str(pack_root),
        "lab_cache": str(lab_cache),
        "files": files,
        "provider_requested": provider,
    }

    manifest_path = lab_cache / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest
