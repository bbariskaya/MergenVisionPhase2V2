"""Synthetic tests for ByteTrack IoU and Hybrid face trackers."""

from __future__ import annotations

import numpy as np
import pytest

from mergenvision_video_lab.contracts import BBoxXYXY, FaceObservation, Landmarks5, QualityMetrics
from mergenvision_video_lab.tracking.byte_tracker import ByteTrackIoUTracker
from mergenvision_video_lab.tracking.hybrid_face_tracker import HybridFaceByteTracker

LANDMARKS = Landmarks5(
    left_eye=(40.0, 50.0),
    right_eye=(70.0, 50.0),
    nose=(55.0, 70.0),
    left_mouth=(45.0, 90.0),
    right_mouth=(65.0, 90.0),
)


def _unit_vector(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _obs(
    frame_index: int,
    pts_ns: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    score: float,
    quality_score: float = 0.8,
    embedding_index: int | None = None,
    observation_id: str | None = None,
) -> FaceObservation:
    return FaceObservation(
        observation_id=observation_id or f"obs:{frame_index}:{x1}",
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
        detector_score=score,
        landmarks_5=LANDMARKS,
        quality=QualityMetrics(
            bbox_width_px=x2 - x1,
            bbox_height_px=y2 - y1,
            bbox_min_side_px=min(x2 - x1, y2 - y1),
            bbox_area_px=(x2 - x1) * (y2 - y1),
            detector_score=score,
            grayscale_laplacian_variance=100.0,
            brightness_mean=128.0,
            brightness_std=30.0,
            dark_clip_fraction=0.0,
            bright_clip_fraction=0.0,
            interocular_distance_px=30.0,
            alignment_reprojection_error_px=1.0,
            alignment_error_normalized_by_interocular=0.03,
            landmark_geometry_valid=True,
            finite_embedding=embedding_index is not None,
            composite_quality_score=quality_score,
        ),
        tracking_eligible=True,
        recognition_eligible=embedding_index is not None,
        embedding_index=embedding_index,
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


def test_empty_frame_does_not_create_tracks(byte_config: dict) -> None:
    tracker = ByteTrackIoUTracker(byte_config)
    embeddings = np.zeros((0, 512), dtype=np.float32)
    assignments = tracker.update(0, 0, [], embeddings, scene_cut_before=False)
    assert assignments == []
    tracker.finalize()
    assert len(tracker.removed_tracklets()) == 0


def test_single_stable_face_creates_one_track(byte_config: dict) -> None:
    tracker = ByteTrackIoUTracker(byte_config)
    embeddings = np.zeros((0, 512), dtype=np.float32)
    obs = _obs(0, 0, 100.0, 100.0, 150.0, 150.0, 0.8)
    assignments = tracker.update(0, 0, [obs], embeddings, scene_cut_before=False)
    assert len(assignments) == 1
    rt_id = assignments[0].raw_tracklet_id
    # Next frame nearby -> same track.
    obs2 = _obs(1, 33_000_000, 102.0, 102.0, 152.0, 152.0, 0.8)
    assignments2 = tracker.update(1, 33_000_000, [obs2], embeddings, scene_cut_before=False)
    assert len(assignments2) == 1
    assert assignments2[0].raw_tracklet_id == rt_id


def test_two_simultaneous_faces_get_distinct_tracks(byte_config: dict) -> None:
    tracker = ByteTrackIoUTracker(byte_config)
    embeddings = np.zeros((0, 512), dtype=np.float32)
    obs1 = _obs(0, 0, 100.0, 100.0, 150.0, 150.0, 0.8)
    obs2 = _obs(0, 0, 300.0, 100.0, 350.0, 150.0, 0.8)
    assignments = tracker.update(0, 0, [obs1, obs2], embeddings, scene_cut_before=False)
    assert len(assignments) == 2
    assert assignments[0].raw_tracklet_id != assignments[1].raw_tracklet_id


def test_temporary_occlusion_keeps_track(byte_config: dict) -> None:
    tracker = ByteTrackIoUTracker(byte_config)
    embeddings = np.zeros((0, 512), dtype=np.float32)
    obs = _obs(0, 0, 100.0, 100.0, 150.0, 150.0, 0.8)
    rt_id = tracker.update(0, 0, [obs], embeddings, scene_cut_before=False)[0].raw_tracklet_id
    # Face missing for a few frames.
    for i in range(1, 4):
        tracker.update(i, i * 33_000_000, [], embeddings, scene_cut_before=False)
    obs2 = _obs(4, 4 * 33_000_000, 105.0, 105.0, 155.0, 155.0, 0.8)
    assignments = tracker.update(4, 4 * 33_000_000, [obs2], embeddings, scene_cut_before=False)
    assert assignments[0].raw_tracklet_id == rt_id


def test_lost_timeout_creates_new_tracklet(byte_config: dict) -> None:
    tracker = ByteTrackIoUTracker(byte_config)
    embeddings = np.zeros((0, 512), dtype=np.float32)
    obs = _obs(0, 0, 100.0, 100.0, 150.0, 150.0, 0.8)
    first_id = tracker.update(0, 0, [obs], embeddings, scene_cut_before=False)[0].raw_tracklet_id
    # Gap exceeds max_lost_frames.
    for i in range(1, 40):
        tracker.update(i, i * 33_000_000, [], embeddings, scene_cut_before=False)
    obs2 = _obs(40, 40 * 33_000_000, 105.0, 105.0, 155.0, 155.0, 0.8)
    assignments = tracker.update(40, 40 * 33_000_000, [obs2], embeddings, scene_cut_before=False)
    assert assignments[0].raw_tracklet_id != first_id


def test_scene_cut_creates_new_tracklet(byte_config: dict) -> None:
    tracker = ByteTrackIoUTracker(byte_config)
    embeddings = np.zeros((0, 512), dtype=np.float32)
    obs = _obs(0, 0, 100.0, 100.0, 150.0, 150.0, 0.8)
    first_id = tracker.update(0, 0, [obs], embeddings, scene_cut_before=False)[0].raw_tracklet_id
    obs2 = _obs(1, 33_000_000, 105.0, 105.0, 155.0, 155.0, 0.8)
    assignments = tracker.update(1, 33_000_000, [obs2], embeddings, scene_cut_before=True)
    assert assignments[0].raw_tracklet_id != first_id


def test_low_score_second_stage_recovery(byte_config: dict) -> None:
    tracker = ByteTrackIoUTracker(byte_config)
    embeddings = np.zeros((0, 512), dtype=np.float32)
    obs_high = _obs(0, 0, 100.0, 100.0, 150.0, 150.0, 0.8)
    rt_id = tracker.update(0, 0, [obs_high], embeddings, scene_cut_before=False)[0].raw_tracklet_id
    # Low-score detection in next frame should recover the track.
    obs_low = _obs(1, 33_000_000, 102.0, 102.0, 152.0, 152.0, 0.3)
    assignments = tracker.update(1, 33_000_000, [obs_low], embeddings, scene_cut_before=False)
    assert len(assignments) == 1
    assert assignments[0].raw_tracklet_id == rt_id


def test_no_duplicate_track_ids_in_one_frame(byte_config: dict) -> None:
    tracker = ByteTrackIoUTracker(byte_config)
    embeddings = np.zeros((0, 512), dtype=np.float32)
    obs1 = _obs(0, 0, 100.0, 100.0, 150.0, 150.0, 0.8)
    obs2 = _obs(0, 0, 102.0, 102.0, 152.0, 152.0, 0.8)
    assignments = tracker.update(0, 0, [obs1, obs2], embeddings, scene_cut_before=False)
    ids = [a.raw_tracklet_id for a in assignments]
    assert len(ids) == len(set(ids))


def test_frame_index_regression_raises(byte_config: dict) -> None:
    tracker = ByteTrackIoUTracker(byte_config)
    embeddings = np.zeros((0, 512), dtype=np.float32)
    tracker.update(1, 33_000_000, [], embeddings, False)
    with pytest.raises(ValueError):
        tracker.update(0, 0, [], embeddings, False)


def test_hybrid_appearance_fusion_prefers_similar_embedding(byte_config: dict) -> None:
    config = dict(byte_config)
    embeddings = np.zeros((3, 512), dtype=np.float32)
    embeddings[0] = _unit_vector(1)
    embeddings[1] = _unit_vector(2)
    embeddings[2] = embeddings[0]  # identical to first

    tracker = HybridFaceByteTracker(config)
    obs1 = _obs(0, 0, 100.0, 100.0, 150.0, 150.0, 0.8, embedding_index=0)
    rt_id = tracker.update(0, 0, [obs1], embeddings, False)[0].raw_tracklet_id

    # Two nearby faces, both satisfying the IoU gate.  The one with the same
    # embedding as the existing track should win despite having lower IoU than
    # the different-embedding detection.
    obs_diff = _obs(1, 33_000_000, 102.0, 102.0, 152.0, 152.0, 0.8, embedding_index=1)
    obs_same = _obs(1, 33_000_000, 104.0, 104.0, 154.0, 154.0, 0.8, embedding_index=2)
    assignments = tracker.update(1, 33_000_000, [obs_diff, obs_same], embeddings, False)
    by_obs = {a.observation_id: a.raw_tracklet_id for a in assignments}
    assert by_obs[obs_same.observation_id] == rt_id
    assert by_obs[obs_diff.observation_id] != rt_id
