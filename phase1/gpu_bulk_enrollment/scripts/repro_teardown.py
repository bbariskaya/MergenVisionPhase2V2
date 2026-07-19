"""M4 reproducer: repeated create/warmup/infer/close of GpuFacePipeline.

Run inside the pinned ``mv-phase1-bulk`` GPU container.  Exits non-zero if any
iteration raises or if the process segfaults during teardown.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from cuda.bindings import runtime as cuda_runtime

from mv_phase1_bulk.config import load_model_profile
from mv_phase1_bulk.pipeline import GpuFacePipeline


_ORIG_INIT = GpuFacePipeline.__init__
_SHARED_STREAMS: dict[int, int] = {}


def _patched_init(self: GpuFacePipeline, *, model_profile: dict[str, Any], device_id: int = 0) -> None:
    # Run the original constructor, then replace its per-pipeline stream with a
    # shared per-device stream so repeated create/close cycles do not leak handles.
    _ORIG_INIT(self, model_profile=model_profile, device_id=device_id)
    if device_id in _SHARED_STREAMS:
        with contextlib.suppress(Exception):
            cuda_runtime.cudaStreamDestroy(self._stream)
        self._stream = _SHARED_STREAMS[device_id]
    else:
        _SHARED_STREAMS[device_id] = self._stream


def _patched_close(self: GpuFacePipeline) -> None:
    if getattr(self, "_closed", False):
        return
    self._closed = True
    with contextlib.suppress(Exception):
        cuda_runtime.cudaSetDevice(self._device_id)
    if hasattr(self, "_stream"):
        with contextlib.suppress(Exception):
            cuda_runtime.cudaStreamSynchronize(self._stream)
    for obj in (
        self._decoder,
        self._preprocessor,
        self._postprocess,
        self._aligner,
        self._recognizer,
        self._detector_engine,
        self._arena,
    ):
        with contextlib.suppress(Exception):
            obj.close()
    # Force decoder/engine objects to release CUDA handles before the stream is destroyed.
    self._decoder = None
    self._preprocessor = None
    self._detector_engine = None
    self._postprocess = None
    self._aligner = None
    self._recognizer = None
    self._arena = None
    # The shared stream is intentionally left alive; destroying it while CV-CUDA
    # may cache the handle causes cudaErrorInvalidResourceHandle on the next run.
    if hasattr(self, "_stream"):
        del self._stream


def _patched_del(self: GpuFacePipeline) -> None:
    if not getattr(self, "_closed", False):
        with contextlib.suppress(Exception):
            self.close()


def _gpu_memory_mb(device_id: int = 0) -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu=memory.used",
                f"--id={device_id}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="config/model_profile.json", type=Path)
    parser.add_argument("--image", default="lfw/Jessica_Capshaw/Jessica_Capshaw_0001.jpg", type=Path)
    parser.add_argument("--iterations", default=30, type=int)
    parser.add_argument("--device-id", default=0, type=int)
    parser.add_argument("--fix", action="store_true", help="Apply experimental close-order patch")
    args = parser.parse_args()

    if args.fix:
        GpuFacePipeline.__init__ = _patched_init
        GpuFacePipeline.close = _patched_close
        GpuFacePipeline.__del__ = _patched_del
        print("EXPERIMENTAL close-order + shared-stream patch enabled", flush=True)

    if not args.image.exists():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 2

    profile_path = args.profile.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    model_profile = load_model_profile(profile_path, repo_root=repo_root)
    image_bytes = args.image.read_bytes()

    baseline_memory = _gpu_memory_mb(args.device_id)
    print(f"baseline GPU memory used: {baseline_memory} MiB")

    failures = 0
    peak_memory = baseline_memory
    start = time.perf_counter()

    for iteration in range(1, args.iterations + 1):
        pipeline: GpuFacePipeline | None = None
        results = None
        try:
            pipeline = GpuFacePipeline(
                model_profile=model_profile,
                device_id=args.device_id,
            )
            pipeline.warmup()
            results = pipeline.extract_batch([image_bytes])
            assert len(results) == 1
            result = results[0]
            face_count = len(result.faces)
            status = result.status
            reason = result.rejection_reason
            # Release device tensors before closing the pipeline so arenas can be freed safely.
            results = None
            del result
            print(
                f"iter {iteration:02d}: status={status} faces={face_count} "
                f"reason={reason}",
                flush=True,
            )

            used = _gpu_memory_mb(args.device_id)
            if used is not None and used > (peak_memory or 0):
                peak_memory = used
        except Exception as exc:
            failures += 1
            print(f"iter {iteration:02d}: FAILURE {exc}", file=sys.stderr, flush=True)
        finally:
            if results is not None:
                del results
            if pipeline is not None:
                try:
                    pipeline.close()
                    print(f"iter {iteration:02d}: close ok", flush=True)
                except Exception as exc:
                    print(f"iter {iteration:02d}: close failed {exc}", file=sys.stderr, flush=True)
                    failures += 1
                # Deliberately exercise destructor idempotency.
                del pipeline
                gc.collect()
                used_after = _gpu_memory_mb(args.device_id)
                print(f"iter {iteration:02d}: after close memory={used_after} MiB", flush=True)

    elapsed = time.perf_counter() - start
    final_memory = _gpu_memory_mb(args.device_id)

    print(f"iterations={args.iterations} failures={failures}", flush=True)
    print(f"elapsed={elapsed:.2f}s avg={elapsed/args.iterations:.3f}s", flush=True)
    print(f"baseline={baseline_memory} MiB peak={peak_memory} MiB final={final_memory} MiB", flush=True)

    if failures:
        return 1
    if final_memory is not None and baseline_memory is not None and final_memory > baseline_memory + 200:
        print(f"ERROR: GPU memory leaked >200 MiB (baseline={baseline_memory} final={final_memory})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
