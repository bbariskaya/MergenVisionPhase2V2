#!/usr/bin/env python3
"""Build TensorRT engines from ONNX models and update the engine manifest.

This script is intended to run inside the pinned NVIDIA TensorRT container where
`trtexec` and the target CUDA/TensorRT libraries are present. It never downloads
models; it only builds engines from the ONNX files already in the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _tensorrt_version() -> str:
    try:
        import tensorrt as trt

        return ".".join(str(x) for x in trt.__version_info__)
    except Exception:
        out = _run(["trtexec", "--version"], check=False)
        for line in (out.stdout + out.stderr).splitlines():
            if "TensorRT version" in line:
                return line.split(":")[-1].strip()
        return "unknown"


def _cuda_version() -> str:
    out = _run(["nvcc", "--version"], check=False)
    for line in (out.stdout + out.stderr).splitlines():
        if "release" in line:
            start = line.find("release") + len("release")
            return line[start:].split(",")[0].strip()
    return "unknown"


def _gpu_info() -> tuple[str, str]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,compute_cap",
            "--format=csv,noheader",
        ],
        check=False,
    )
    if out.returncode != 0 or not out.stdout.strip():
        return ("unknown", "unknown")
    uuid, compute_cap = out.stdout.strip().split(",", 1)
    return uuid.strip(), compute_cap.strip()


def _shape_arg(input_name: str, shape: list[int]) -> str:
    return f"{input_name}:{ 'x'.join(str(x) for x in shape)}"


def _build_engine(
    onnx_path: pathlib.Path,
    engine_path: pathlib.Path,
    min_shape: list[int],
    opt_shape: list[int],
    max_shape: list[int],
    input_name: str,
    precision: str,
) -> None:
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    shapes = [
        f"--minShapes={_shape_arg(input_name, min_shape)}",
        f"--optShapes={_shape_arg(input_name, opt_shape)}",
        f"--maxShapes={_shape_arg(input_name, max_shape)}",
    ]
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--{precision}",
        *shapes,
    ]
    _run(cmd)


def _find_repo_root(profile_path: pathlib.Path) -> pathlib.Path:
    start = profile_path.resolve().parent
    for candidate in [start, *start.parents]:
        if (candidate / "backend" / "artifacts" / "models").is_dir():
            return candidate
    cwd = pathlib.Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "backend" / "artifacts" / "models").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate repository root containing backend/artifacts/models"
    )


def _engine_output_path(
    repo_root: pathlib.Path,
    engines_dir: pathlib.Path | None,
    manifest_entry: dict[str, Any],
    onnx_path: pathlib.Path,
) -> pathlib.Path:
    if engines_dir is not None:
        engines_dir.mkdir(parents=True, exist_ok=True)
        default_engine_name = onnx_path.stem + ".engine"
        return engines_dir / manifest_entry.get("engine_path", default_engine_name).replace("/", "_")
    engine_rel = pathlib.Path(
        manifest_entry.get("engine_path", str(onnx_path.with_suffix(".engine").relative_to(repo_root)))
    )
    return repo_root / engine_rel


def _manifest_engine_path(repo_root: pathlib.Path, engine_path: pathlib.Path) -> str:
    try:
        return str(engine_path.relative_to(repo_root))
    except ValueError:
        return str(engine_path)


def build(profile_path: pathlib.Path, engines_dir: pathlib.Path | None) -> None:
    with profile_path.open() as fh:
        raw: dict[str, Any] = json.load(fh)

    repo_root = _find_repo_root(profile_path)

    manifest: dict[str, Any] = raw.setdefault("engine_manifest", {})
    container_digest = manifest.get(
        "container_digest",
        os.environ.get(
            "TENSORRT_IMAGE",
            "nvcr.io/nvidia/tensorrt:26.03-py3@sha256:ade1b30517b3d66b911a3cd7faf0146484ab8956098abe66b96b944fa36f4861",
        ),
    )

    for model_key, spec in raw["models"].items():
        onnx_path = repo_root / spec["onnx_path"]
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        actual_onnx_sha = _sha256(onnx_path)
        expected_onnx_sha = spec.get("onnx_sha256")
        if expected_onnx_sha and actual_onnx_sha != expected_onnx_sha:
            raise RuntimeError(
                f"ONNX SHA mismatch for {model_key}: expected {expected_onnx_sha}, got {actual_onnx_sha}"
            )

    engine_entries: dict[str, Any] = {}
    for model_key in ("retinaface_r50_dynamic", "glintr100"):
        spec = raw["models"][model_key]
        onnx_path = repo_root / spec["onnx_path"]
        entry = manifest.get(model_key, {})
        onnx_path = repo_root / spec["onnx_path"]
        engine_path = _engine_output_path(repo_root, engines_dir, entry, onnx_path)

        if model_key == "retinaface_r50_dynamic":
            sub = raw["detector"]
            input_name = sub["input_tensor_name"]
            precision = entry.get("precision", "fp16")
            profile = sub["dynamic_profile"]
        elif model_key == "glintr100":
            sub = raw["recognizer"]
            input_name = sub["input_tensor_name"]
            precision = entry.get("precision", "fp16")
            profile = sub["dynamic_profile"]
        else:
            raise ValueError(f"unknown model key: {model_key}")

        build_command = (
            f"trtexec --onnx={onnx_path} --saveEngine={engine_path} "
            f"--{precision} "
            f"--minShapes={_shape_arg(input_name, profile['min'])} "
            f"--optShapes={_shape_arg(input_name, profile['opt'])} "
            f"--maxShapes={_shape_arg(input_name, profile['max'])}"
        )

        _build_engine(
            onnx_path,
            engine_path,
            profile["min"],
            profile["opt"],
            profile["max"],
            input_name,
            precision,
        )

        engine_sha = _sha256(engine_path)
        engine_entries[model_key] = {
            "engine_path": _manifest_engine_path(repo_root, engine_path),
            "engine_sha256": engine_sha,
            "profile": profile,
            "precision": precision,
            "build_command": build_command,
        }

    gpu_uuid, gpu_compute = _gpu_info()
    manifest.update(engine_entries)
    manifest.update(
        {
            "build_command": build_command,
            "tensorrt_version": _tensorrt_version(),
            "cuda_version": _cuda_version(),
            "container_digest": container_digest,
            "gpu_compute_capability": gpu_compute,
            "gpu_uuid": gpu_uuid,
            "build_timestamp": datetime.now(UTC).isoformat(),
        }
    )

    with profile_path.open("w") as fh:
        json.dump(raw, fh, indent=2)
        fh.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build TensorRT engines and update manifest")
    parser.add_argument(
        "--profile",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parents[2]
        / "config"
        / "model_profiles"
        / "retinaface_r50_glintr100_v1.example.json",
    )
    parser.add_argument(
        "--engines-dir",
        type=pathlib.Path,
        default=None,
        help="Override directory for engine outputs; defaults to paths listed in the manifest.",
    )
    args = parser.parse_args()

    build(args.profile, args.engines_dir)
    print("Engine build complete; manifest updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
