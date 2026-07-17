#!/usr/bin/env python3
"""Native video runtime gate for the pinned DeepStream container.

This script runs inside `mergenvision/deepstream-dev:9.0` and verifies the
GStreamer/DeepStream/TensorRT runtime plus the engine manifest. It exits
non-zero when any required contract is missing so it can be used as the
`phase2-m5-native-runtime-gate` Makefile target.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

REPO_ROOT = pathlib.Path(os.environ.get("MERGENVISION_REPO_ROOT", "/workspace"))
PROFILE_PATH = REPO_ROOT / "backend" / "config" / "model_profiles" / "retinaface_r50_glintr100_v1_deepstream9.json"

FAILURES: list[str] = []


def _fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"FAIL: {msg}", file=sys.stderr)


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _load_profile() -> dict[str, Any]:
    with PROFILE_PATH.open() as fh:
        return json.load(fh)


def _check_command(name: str, cmd: list[str], expected: list[str] | None = None) -> str | None:
    out = _run(cmd)
    if out.returncode != 0:
        _fail(f"{name} command failed (rc={out.returncode}): {out.stderr.strip()[:200]}")
        return None
    combined = (out.stdout + out.stderr).lower()
    if expected:
        for token in expected:
            if token.lower() not in combined:
                _fail(f"{name} output missing expected token: {token}")
                return None
    return out.stdout + out.stderr


def _check_gstreamer_plugins() -> None:
    text = _check_command(
        "gst-inspect nvv4l2decoder",
        ["gst-inspect-1.0", "nvv4l2decoder"],
        expected=["Factory Details", "NVIDIA v4l2 video decoder", "nvvideo4linux2"],
    )
    if text and "memory:NVMM" not in text:
        _fail("nvv4l2decoder does not advertise NVMM capability")

    text = _check_command(
        "gst-inspect nvstreammux",
        ["bash", "-c", "unset USE_NEW_NVSTREAMMUX && gst-inspect-1.0 nvstreammux"],
        expected=[
            "GstNvStreamMux",
            "sink_%u",
            "batch-size",
            "batched-push-timeout",
            "nvbuf-memory-type",
            "buffer-pool-size",
            "live-source",
        ],
    )
    if text and "libnvdsgst_multistream.so" not in text:
        _fail("nvstreammux factory does not come from nvdsgst_multistream")


def _check_tensorrt() -> None:
    out = _run(["trtexec", "--version"])
    text = out.stdout + out.stderr
    if "TensorRT" not in text:
        _fail(f"trtexec did not emit TensorRT version string (rc={out.returncode})")
        return
    for line in text.splitlines():
        if "TensorRT v" in line:
            print(f"OK: TensorRT line: {line.strip()}")
            if "v101401" not in line and "10.14" not in text:
                _fail(f"TensorRT version is not 10.14.1: {line.strip()}")
            break


def _check_models_and_manifest() -> None:
    if not PROFILE_PATH.exists():
        _fail(f"Profile not found: {PROFILE_PATH}")
        return

    profile = _load_profile()
    manifest = profile.get("engine_manifest", {})
    container_digest = manifest.get("container_digest", "")
    expected_digest = (
        "mergenvision/deepstream-dev:9.0@sha256:"
        "309dce0982d2643a51c3d17aded6d0ebc890c834f3621b439d93496f0d46b616"
    )
    if container_digest != expected_digest:
        _fail(f"Manifest container_digest mismatch: {container_digest}")

    for key in ("retinaface_r50_dynamic", "glintr100"):
        spec = profile["models"][key]
        onnx_path = REPO_ROOT / spec["onnx_path"]
        if not onnx_path.exists():
            _fail(f"ONNX model missing: {onnx_path}")
            continue
        actual_sha = _sha256(onnx_path)
        expected_sha = spec.get("onnx_sha256")
        if expected_sha and actual_sha != expected_sha:
            _fail(f"ONNX SHA mismatch for {key}: expected {expected_sha}, got {actual_sha}")

        entry = manifest.get(key, {})
        engine_path = REPO_ROOT / entry.get("engine_path", "")
        if not engine_path.exists():
            _fail(f"Engine missing: {engine_path}")
            continue
        actual_engine_sha = _sha256(engine_path)
        expected_engine_sha = entry.get("engine_sha256")
        if expected_engine_sha and actual_engine_sha != expected_engine_sha:
            _fail(
                f"Engine SHA mismatch for {key}: expected {expected_engine_sha}, got {actual_engine_sha}"
            )

    for field in ("tensorrt_version", "cuda_version", "gpu_uuid", "build_timestamp"):
        if not manifest.get(field):
            _fail(f"Engine manifest field '{field}' is not populated")


def main() -> int:
    print(f"Runtime gate starting, repo root: {REPO_ROOT}")
    print(f"Profile: {PROFILE_PATH}")

    _run(["nvidia-smi", "-L"])
    _check_gstreamer_plugins()
    _check_tensorrt()
    _check_models_and_manifest()

    if FAILURES:
        print("\n=== Runtime gate FAILED ===", file=sys.stderr)
        for f in FAILURES:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("\n=== Runtime gate PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
