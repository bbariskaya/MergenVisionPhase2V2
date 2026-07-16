"""Benchmark harness for extraction and frozen-metadata replay."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import psutil

from mergenvision_video_lab.contracts import FaceObservation, RunManifest
from mergenvision_video_lab.tracking.byte_tracker import ByteTrackIoUTracker
from mergenvision_video_lab.tracking.hybrid_face_tracker import HybridFaceByteTracker


def _current_rss_mb() -> float:
    return float(psutil.Process().memory_info().rss) / (1024 * 1024)


def benchmark_extraction(manifest: RunManifest) -> dict[str, Any]:
    """Summarize reference extraction performance from the manifest."""
    timing = manifest.extraction_timing
    decoded = manifest.decoded_frame_count
    processed = manifest.processed_frame_count
    total_sec = max(timing.total_seconds, 1e-9)
    return {
        "stage": "reference_extraction",
        "decoded_frames": decoded,
        "face_observations": manifest.observation_count,
        "valid_embeddings": manifest.valid_embedding_count,
        "decode_fps": decoded / max(timing.decode_seconds, 1e-9),
        "oracle_inference_fps": processed / max(timing.oracle_seconds, 1e-9),
        "total_reference_fps": decoded / total_sec,
        "timing_seconds": timing.model_dump(mode="json"),
        "providers_actual": manifest.providers_actual,
    }


def _group_observations_by_frame(
    observations: list[FaceObservation],
    decoded_frame_count: int,
) -> dict[int, list[FaceObservation]]:
    grouped: dict[int, list[FaceObservation]] = {i: [] for i in range(decoded_frame_count)}
    for obs in observations:
        if obs.frame_index in grouped:
            grouped[obs.frame_index].append(obs)
    return grouped


def benchmark_replay(
    observations: list[FaceObservation],
    embeddings: np.ndarray,
    manifest: RunManifest,
    tracker_class: type[ByteTrackIoUTracker] | type[HybridFaceByteTracker],
    tracking_config: dict[str, Any],
    chunk_size: int,
    warmup_runs: int,
    measured_runs: int,
    target_fps: float,
) -> dict[str, Any]:
    """Benchmark one tracker strategy with one chunk size."""
    strategy = tracker_class.strategy
    decoded = manifest.decoded_frame_count
    grouped = _group_observations_by_frame(observations, decoded)
    scene_cut_frames = set(manifest.scene_cut_frame_indices)

    def one_run() -> tuple[list[Any], float, int]:
        tracker = tracker_class(tracking_config)
        max_active = 0
        t0 = time.perf_counter()
        for frame_index in range(decoded):
            frame_obs = grouped.get(frame_index, [])
            # Frame PTS: derive from the first observation or use frame index * 1ms fallback.
            pts_ns = frame_obs[0].pts_ns if frame_obs else frame_index * 1000000
            scene_cut_before = frame_index in scene_cut_frames
            tracker.update(frame_index, pts_ns, frame_obs, embeddings, scene_cut_before)
            active = len(tracker.active_tracklet_ids())
            if active > max_active:
                max_active = active
        tracker.finalize()
        elapsed = time.perf_counter() - t0
        return tracker.removed_tracklets(), elapsed, max_active

    # Warmup.
    for _ in range(warmup_runs):
        one_run()

    times: list[float] = []
    max_active = 0
    final_tracklets: list[Any] = []
    for _ in range(measured_runs):
        tracklets, elapsed, run_max_active = one_run()
        times.append(elapsed)
        final_tracklets = tracklets
        if run_max_active > max_active:
            max_active = run_max_active

    times_ms = [t * 1000.0 / max(decoded, 1) for t in times]
    p50 = float(np.percentile(times_ms, 50))
    p95 = float(np.percentile(times_ms, 95))
    p99 = float(np.percentile(times_ms, 99))
    avg_fps = decoded / max(np.mean(times), 1e-9)

    target_ms = 1000.0 / target_fps
    return {
        "stage": "frozen_metadata_replay",
        "strategy": strategy,
        "chunk_size": chunk_size,
        "decoded_frames": decoded,
        "total_observations": len(observations),
        "measured_runs": measured_runs,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mean_ms": float(np.mean(times_ms)),
        "replay_fps": avg_fps,
        "informational_target_fps": target_fps,
        "target_ms": target_ms,
        "pass_against_target": p50 <= target_ms,
        "peak_rss_mb": _current_rss_mb(),
        "max_active_tracks_estimate": max_active,
        "final_raw_tracklet_count": len(final_tracklets),
    }
