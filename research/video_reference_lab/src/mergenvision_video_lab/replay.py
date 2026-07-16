"""Replay frozen artifacts through a tracker without rerunning inference."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mergenvision_video_lab.artifact_store import ArtifactStore
from mergenvision_video_lab.contracts import FaceObservation, FrameRecord, TrackAssignment
from mergenvision_video_lab.tracking.byte_tracker import ByteTrackIoUTracker


def _group_observations_by_frame(
    observations: list[FaceObservation],
) -> dict[int, list[FaceObservation]]:
    grouped: dict[int, list[FaceObservation]] = defaultdict(list)
    for obs in observations:
        grouped[obs.frame_index].append(obs)
    return grouped


def replay_frames(
    run_dir: Path | str,
    tracking_config: dict[str, Any],
    strategy: str = "byte_iou",
    chunk_size: int = 1,
) -> dict[str, Any]:
    """Replay a frozen reference run through a tracker.

    ``chunk_size`` is an I/O/read-ahead grouping hint only; the tracker state
    never resets at chunk boundaries. The same ordered frame sequence must be
    processed regardless of chunk_size.
    """
    run_dir = Path(run_dir)
    store = ArtifactStore(run_dir)
    store.validate()

    frames = store.read_frames()
    observations = store.read_observations()
    embeddings = store.read_embeddings()

    grouped = _group_observations_by_frame(observations)

    if strategy != "byte_iou":
        raise ValueError(f"unsupported replay strategy: {strategy}")

    tracker = ByteTrackIoUTracker(tracking_config)
    assignments: list[TrackAssignment] = []

    for frame in frames:
        frame_assignments = tracker.update(
            frame.frame_index,
            frame.pts_ns,
            grouped.get(frame.frame_index, []),
            embeddings,
            scene_cut_before=frame.scene_cut_before,
        )
        assignments.extend(frame_assignments)

    tracker.finalize()

    return {
        "strategy": strategy,
        "chunk_size": chunk_size,
        "decoded_frame_count": len(frames),
        "assignment_count": len(assignments),
        "raw_tracklet_count": len(tracker.removed_tracklets()),
        "assignments": [a.model_dump(mode="json") for a in assignments],
        "tracklets": [t.to_summary().model_dump(mode="json") for t in tracker.removed_tracklets()],
    }


def replay_for_chunk_invariance(
    run_dir: Path | str,
    frames: list[FrameRecord],
    observations: list[FaceObservation],
    embeddings: np.ndarray,
    tracking_config: dict[str, Any],
    strategy: str = "byte_iou",
    chunk_size: int = 1,
) -> dict[str, Any]:
    """Replay with already-loaded artifacts for deterministic chunk parity tests."""
    if strategy != "byte_iou":
        raise ValueError(f"unsupported replay strategy: {strategy}")

    grouped = _group_observations_by_frame(observations)
    tracker = ByteTrackIoUTracker(tracking_config)
    assignments: list[TrackAssignment] = []

    for frame in frames:
        frame_assignments = tracker.update(
            frame.frame_index,
            frame.pts_ns,
            grouped.get(frame.frame_index, []),
            embeddings,
            scene_cut_before=frame.scene_cut_before,
        )
        assignments.extend(frame_assignments)

    tracker.finalize()

    return {
        "strategy": strategy,
        "chunk_size": chunk_size,
        "decoded_frame_count": len(frames),
        "assignment_count": len(assignments),
        "raw_tracklet_count": len(tracker.removed_tracklets()),
        "assignments": assignments,
        "tracklets": tracker.removed_tracklets(),
    }
