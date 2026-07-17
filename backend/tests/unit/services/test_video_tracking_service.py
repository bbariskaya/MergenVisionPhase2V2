"""M6 raw tracklet builder and template selector tests."""

from __future__ import annotations

import pytest

from app.application.ports.video_observations import FaceObservation, VideoObservationFrame
from app.application.services.video_tracking_service import VideoTrackingService
from app.domain.value_objects import BoundingBox
from app.infrastructure.uuid7 import generate_uuid7


def _det(
    x: int,
    frame: int = 0,
    *,
    quality_score: float = 0.8,
    tracking_eligible: bool = True,
    recognition_eligible: bool = True,
) -> FaceObservation:
    return FaceObservation(
        detection_id=f"d-{frame}-{x}",
        ordinal=0,
        bbox=BoundingBox(x=x, y=0, width=10, height=10),
        landmarks=(0.0,) * 10,
        detector_score=0.9,
        quality_score=quality_score,
        tracking_eligible=tracking_eligible,
        recognition_eligible=recognition_eligible,
        rejection_code="",
        embedding=(0.0,) * 512,
        model_version="retinaface_r50_glintr100_v1",
        preprocess_version="cuda_nv12_align_v1",
    )


def _frame(
    frame_index: int,
    detections: tuple[FaceObservation, ...],
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
        detections=detections,
    )


def test_service_can_be_constructed() -> None:
    svc = VideoTrackingService(max_gap_frames=5, iou_threshold=0.3)
    assert svc.max_gap_frames == 5
    assert svc.iou_threshold == 0.3


def test_single_detection_per_frame_yields_one_tracklet() -> None:
    svc = VideoTrackingService(max_gap_frames=3, iou_threshold=0.3)
    frames = [_frame(i, (_det(i, i),)) for i in range(5)]
    tracklets = svc.build_raw_tracklets(frames)
    assert len(tracklets) == 1
    assert len(tracklets[0].detections) == 5
    assert tracklets[0].state == "lost"


def test_two_separated_objects_yield_two_tracklets() -> None:
    svc = VideoTrackingService(max_gap_frames=3, iou_threshold=0.3)
    frames = [
        _frame(
            i,
            (
                _det(10, i),
                _det(200, i),
            ),
        )
        for i in range(3)
    ]
    tracklets = svc.build_raw_tracklets(frames)
    assert len(tracklets) == 2
    assert all(len(t.detections) == 3 for t in tracklets)


def test_gap_within_max_gap_keeps_one_tracklet() -> None:
    svc = VideoTrackingService(max_gap_frames=2, iou_threshold=0.3)
    frames = [_frame(i, (_det(100, i),)) if i in (0, 3) else _frame(i, ()) for i in range(4)]
    tracklets = svc.build_raw_tracklets(frames)
    assert len(tracklets) == 1


def test_gap_beyond_max_gap_splits_tracklet() -> None:
    svc = VideoTrackingService(max_gap_frames=1, iou_threshold=0.3)
    frames = [_frame(i, (_det(100, i),)) if i in (0, 3) else _frame(i, ()) for i in range(4)]
    tracklets = svc.build_raw_tracklets(frames)
    assert len(tracklets) == 2


def test_tracking_ineligible_detections_are_ignored() -> None:
    svc = VideoTrackingService(max_gap_frames=3, iou_threshold=0.3)
    frames = [_frame(i, (_det(100, i, tracking_eligible=False),)) for i in range(3)]
    tracklets = svc.build_raw_tracklets(frames)
    assert len(tracklets) == 0


def test_job_id_is_carried_through_tracklets() -> None:
    svc = VideoTrackingService(max_gap_frames=3, iou_threshold=0.3)
    job_id = generate_uuid7()
    frames = [_frame(i, (_det(10 * i, i),)) for i in range(3)]
    tracklets = svc.build_raw_tracklets(frames, job_id=job_id)
    assert all(t.job_id == job_id for t in tracklets)


def test_tracklet_quality_aggregates() -> None:
    svc = VideoTrackingService(max_gap_frames=3, iou_threshold=0.3)
    frames = [
        _frame(0, (_det(0, 0, quality_score=0.3),)),
        _frame(1, (_det(0, 1, quality_score=0.5),)),
        _frame(2, (_det(0, 2, quality_score=0.7),)),
    ]
    tracklets = svc.build_raw_tracklets(frames)
    assert len(tracklets) == 1
    assert tracklets[0].mean_quality == pytest.approx(0.5)
    assert tracklets[0].max_quality == pytest.approx(0.7)


def test_template_selects_best_quality_detections() -> None:
    svc = VideoTrackingService(max_gap_frames=3, iou_threshold=0.3)
    low = _det(0, 0, quality_score=0.3)
    high = _det(0, 1, quality_score=0.95)
    frames = [
        _frame(0, (low,)),
        _frame(1, (high,)),
    ]
    tracklets = svc.build_raw_tracklets(frames)
    template = svc.select_tracklet_template(tracklets[0], max_samples=1)
    assert template.sample_indices == [1]
    assert template.qualities == [pytest.approx(0.95)]


def test_template_uses_min_frame_gap_for_diversity() -> None:
    svc = VideoTrackingService(max_gap_frames=3, iou_threshold=0.3)
    d0 = _det(0, 0, quality_score=0.9)
    d1 = _det(0, 2, quality_score=0.95)
    d2 = _det(0, 4, quality_score=0.92)
    frames = [
        _frame(0, (d0,)),
        _frame(2, (d1,)),
        _frame(4, (d2,)),
    ]
    tracklets = svc.build_raw_tracklets(frames)
    template = svc.select_tracklet_template(tracklets[0], max_samples=2, min_frame_gap=2)
    assert len(template.sample_indices) == 2
    selected_frames = [tracklets[0].detections[i].frame_index for i in template.sample_indices]
    assert abs(selected_frames[0] - selected_frames[1]) >= 2


def test_template_returns_sorted_indices_by_frame() -> None:
    svc = VideoTrackingService(max_gap_frames=3, iou_threshold=0.3)
    d0 = _det(0, 0, quality_score=0.5)
    d1 = _det(0, 1, quality_score=0.9)
    frames = [
        _frame(0, (d0,)),
        _frame(1, (d1,)),
    ]
    tracklets = svc.build_raw_tracklets(frames)
    template = svc.select_tracklet_template(tracklets[0], max_samples=2)
    assert template.sample_indices == sorted(template.sample_indices)
