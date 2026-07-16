"""Chunk-invariance tests for the ByteTrack replay pipeline.

Chunk boundaries are I/O partitioning only; they must never reset tracker state.
The same ordered sequence of frames must produce byte-identical assignments and
raw-tracklet summaries regardless of how observations were grouped on disk.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from mergenvision_video_lab.contracts import BBoxXYXY, FaceObservation, Landmarks5, QualityMetrics
from mergenvision_video_lab.tracking.byte_tracker import ByteTrackIoUTracker, Tracklet

LANDMARKS = Landmarks5(
    left_eye=(40.0, 50.0),
    right_eye=(70.0, 50.0),
    nose=(55.0, 70.0),
    left_mouth=(45.0, 90.0),
    right_mouth=(65.0, 90.0),
)


def _obs(
    frame_index: int, pts_ns: int, x1: float, y1: float, x2: float, y2: float
) -> FaceObservation:
    return FaceObservation(
        observation_id=f"obs:{frame_index}:{x1}",
        source_id="synthetic",
        frame_index=frame_index,
        pts=pts_ns // 1_000_000,
        time_base_num=1,
        time_base_den=1_000_000,
        pts_ns=pts_ns,
        frame_width=640,
        frame_height=480,
        detection_ordinal=0,
        bbox_xyxy=BBoxXYXY(x1=x1, y1=y1, x2=x2, y2=y2),
        detector_score=0.8,
        landmarks_5=LANDMARKS,
        quality=QualityMetrics(
            bbox_width_px=x2 - x1,
            bbox_height_px=y2 - y1,
            bbox_min_side_px=min(x2 - x1, y2 - y1),
            bbox_area_px=(x2 - x1) * (y2 - y1),
            detector_score=0.8,
            grayscale_laplacian_variance=100.0,
            brightness_mean=128.0,
            brightness_std=30.0,
            dark_clip_fraction=0.0,
            bright_clip_fraction=0.0,
            interocular_distance_px=30.0,
            alignment_reprojection_error_px=1.0,
            alignment_error_normalized_by_interocular=0.03,
            landmark_geometry_valid=True,
            finite_embedding=False,
            composite_quality_score=0.8,
        ),
        tracking_eligible=True,
        recognition_eligible=False,
        embedding_index=None,
    )


@pytest.fixture
def byte_config() -> dict:
    return {
        "high_detection_threshold": 0.60,
        "low_detection_threshold": 0.10,
        "new_track_threshold": 0.70,
        "first_stage_min_iou": 0.10,
        "second_stage_min_iou": 0.30,
        "unconfirmed_min_iou": 0.30,
        "short_term_min_cosine": 0.10,
        "appearance_weight": 0.35,
        "max_lost_frames": 30,
        "max_lost_ns": 1_000_000_000,
        "scene_cut_reset": True,
        "evidence_top_k": 5,
        "evidence_min_separation_ns": 200_000_000,
    }


@pytest.fixture
def observations() -> list[FaceObservation]:
    """20-frame synthetic sequence with one moving face."""
    return [_obs(i, i * 33_000_000, 100.0 + i, 100.0 + i, 150.0 + i, 150.0 + i) for i in range(20)]


def _run_once(
    byte_config: dict,
    observations: list[FaceObservation],
    chunk_size: int,
) -> tuple[list[tuple[str, str]], list[dict]]:
    """Replay observations through a fresh tracker using the given chunk size.

    The chunk_size only affects how observations are grouped before being fed
    into the tracker; the tracker still receives frames in strict order.
    """
    Tracklet.reset_id_counter()
    tracker = ByteTrackIoUTracker(byte_config)
    embeddings = np.zeros((0, 512), dtype=np.float32)
    grouped: dict[int, list[FaceObservation]] = {}
    for obs in observations:
        grouped.setdefault(obs.frame_index, []).append(obs)

    assignments: list[tuple[str, str]] = []
    for frame_index in range(20):
        # Chunk boundaries must not reset state; we still call update once per
        # ordered frame with the observations that belong to that frame.
        frame_obs = grouped.get(frame_index, [])
        for a in tracker.update(
            frame_index,
            frame_index * 33_000_000,
            frame_obs,
            embeddings,
            scene_cut_before=False,
        ):
            assignments.append((a.observation_id, a.raw_tracklet_id))

    tracker.finalize()

    def _tracklet_to_dict(t: Tracklet) -> dict:
        return {
            "raw_tracklet_id": t.raw_tracklet_id,
            "strategy": t.strategy,
            "state": int(t.state),
            "observation_ids": t.observation_ids,
            "frame_indices": t.frame_indices,
            "pts_ns_list": t.pts_ns_list,
            "bbox_xyxy_history": t.bbox_xyxy_history,
            "scores": t.scores,
            "start_frame_index": t.start_frame_index,
            "start_pts_ns": t.start_pts_ns,
            "last_frame_index": t.last_frame_index,
            "last_pts_ns": t.last_pts_ns,
        }

    summaries = [_tracklet_to_dict(t) for t in tracker.removed_tracklets()]
    return sorted(assignments), summaries


def _hash_object(obj: object) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_chunk_invariance_assignments_and_tracklets(
    byte_config: dict, observations: list[FaceObservation]
) -> None:
    """Assignments and raw tracklets must be identical across chunk sizes."""
    reference_assignments, reference_summaries = _run_once(byte_config, observations, 1)
    reference_summary_hash = _hash_object(reference_summaries)
    reference_assignment_hash = _hash_object(reference_assignments)

    for chunk_size in (8, 17, 64):
        for repetition in range(2):
            assignments, summaries = _run_once(byte_config, observations, chunk_size)
            assert assignments == reference_assignments, (
                f"chunk={chunk_size} repetition={repetition} assignments diverged"
            )
            assert _hash_object(summaries) == reference_summary_hash, (
                f"chunk={chunk_size} repetition={repetition} tracklet summaries diverged"
            )

    # Surface hashes as evidence.
    assert reference_assignment_hash
    assert reference_summary_hash
