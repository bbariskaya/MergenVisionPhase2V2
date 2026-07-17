"""M6 conservative raw-track reconciliation tests."""

from __future__ import annotations

import pytest

from app.application.ports.video_observations import FaceObservation, VideoObservationFrame
from app.application.services.video_reconciliation_service import VideoReconciliationService
from app.application.services.video_tracking_service import VideoTrackingService
from app.domain.value_objects import BoundingBox
from app.infrastructure.uuid7 import generate_uuid7


def _face(
    value: float,
    frame_index: int,
    x: int,
    *,
    quality_score: float = 0.8,
) -> FaceObservation:
    emb = [0.0] * 512
    emb[0] = value
    return FaceObservation(
        detection_id=f"d-{frame_index}-{x}",
        ordinal=0,
        bbox=BoundingBox(x=x, y=0, width=20, height=20),
        landmarks=(0.0,) * 10,
        detector_score=0.9,
        quality_score=quality_score,
        tracking_eligible=True,
        recognition_eligible=True,
        rejection_code="",
        embedding=tuple(emb),
        model_version="retinaface_r50_glintr100_v1",
        preprocess_version="cuda_nv12_align_v1",
    )


def _frame(
    frame_index: int,
    *detections: FaceObservation,
) -> VideoObservationFrame:
    return VideoObservationFrame(
        job_id=str(generate_uuid7()),
        video_id=str(generate_uuid7()),
        stream_index=0,
        frame_index=frame_index,
        source_pts=frame_index,
        pts_ns=frame_index * 33_000_000,
        display_width=640,
        display_height=480,
        detections=tuple(detections),
    )


def _build_tracklets(*frame_groups: list[VideoObservationFrame]) -> list:
    tracker = VideoTrackingService(max_gap_frames=1, iou_threshold=0.3)
    return tracker.build_raw_tracklets([
        frame for group in frame_groups for frame in group
    ])


def test_two_similar_non_overlapping_tracklets_merge() -> None:
    group_a = [_frame(i, _face(1.0, i, x=10)) for i in range(3)]
    group_b = [_frame(i, _face(1.0, i, x=10)) for i in range(5, 8)]
    tracklets = _build_tracklets(group_a, group_b)
    assert len(tracklets) == 2

    reconciler = VideoReconciliationService(merge_threshold=0.9)
    tracks = reconciler.reconcile_tracklets(tracklets)
    assert len(tracks) == 1
    assert len(tracks[0].tracklets) == 2


def test_overlapping_tracklets_do_not_merge() -> None:
    group = [
        _frame(
            i,
            _face(1.0, i, x=10),
            _face(-1.0, i, x=200),
        )
        for i in range(3)
    ]
    tracklets = _build_tracklets(group)
    assert len(tracklets) == 2

    reconciler = VideoReconciliationService(merge_threshold=0.9)
    tracks = reconciler.reconcile_tracklets(tracklets)
    assert len(tracks) == 2


def test_dissimilar_tracklets_do_not_merge() -> None:
    group_a = [_frame(i, _face(1.0, i, x=10)) for i in range(3)]
    group_b = [_frame(i, _face(-1.0, i, x=10)) for i in range(5, 8)]
    tracklets = _build_tracklets(group_a, group_b)
    assert len(tracklets) == 2

    reconciler = VideoReconciliationService(merge_threshold=0.9)
    tracks = reconciler.reconcile_tracklets(tracklets)
    assert len(tracks) == 2


def test_appearance_intervals_cover_merged_tracklets() -> None:
    group_a = [_frame(i, _face(1.0, i, x=10)) for i in range(3)]
    group_b = [_frame(i, _face(1.0, i, x=10)) for i in range(5, 8)]
    tracklets = _build_tracklets(group_a, group_b)
    reconciler = VideoReconciliationService(merge_threshold=0.9)
    tracks = reconciler.reconcile_tracklets(tracklets)
    assert len(tracks) == 1
    assert len(tracks[0].appearance_intervals) == 2


def test_empty_tracklets_yield_empty_tracks() -> None:
    reconciler = VideoReconciliationService(merge_threshold=0.9)
    tracks = reconciler.reconcile_tracklets([])
    assert tracks == []


def test_track_representative_embedding_is_mean_of_detections() -> None:
    group = [_frame(i, _face(1.0, i, x=10)) for i in range(3)]
    tracklets = _build_tracklets(group)
    reconciler = VideoReconciliationService(merge_threshold=0.6)
    tracks = reconciler.reconcile_tracklets(tracklets)
    assert len(tracks) == 1
    assert pytest.approx(tracks[0].representative_embedding[0]) == 1.0
