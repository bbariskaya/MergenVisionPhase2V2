"""Unit tests for data contracts."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from mergenvision_video_lab.contracts import (
    LANDMARK_ORDER,
    BBoxXYXY,
    FaceObservation,
    GroundTruth,
    Landmarks5,
    QualityMetrics,
)


def test_bbox_valid() -> None:
    box = BBoxXYXY(x1=10.0, y1=20.0, x2=30.0, y2=50.0)
    assert box.to_list() == [10.0, 20.0, 30.0, 50.0]


def test_bbox_rejects_zero_area() -> None:
    with pytest.raises(ValidationError):
        BBoxXYXY(x1=10.0, y1=20.0, x2=10.0, y2=50.0)


def test_bbox_rejects_non_finite() -> None:
    with pytest.raises(ValidationError):
        BBoxXYXY(x1=float("nan"), y1=0.0, x2=1.0, y2=1.0)


def test_landmarks_order_exact() -> None:
    lm = Landmarks5(
        left_eye=(1.0, 2.0),
        right_eye=(3.0, 4.0),
        nose=(5.0, 6.0),
        left_mouth=(7.0, 8.0),
        right_mouth=(9.0, 10.0),
    )
    arr = lm.to_array()
    assert arr.shape == (5, 2)
    assert arr.dtype == np.float32


def test_landmark_order_constant() -> None:
    assert LANDMARK_ORDER == [
        "left_eye",
        "right_eye",
        "nose",
        "left_mouth",
        "right_mouth",
    ]


def test_quality_metrics_defaults() -> None:
    q = QualityMetrics(
        bbox_width_px=100.0,
        bbox_height_px=120.0,
        bbox_min_side_px=100.0,
        bbox_area_px=12000.0,
        detector_score=0.9,
        grayscale_laplacian_variance=100.0,
        brightness_mean=128.0,
        brightness_std=30.0,
        dark_clip_fraction=0.0,
        bright_clip_fraction=0.0,
        interocular_distance_px=50.0,
        alignment_reprojection_error_px=2.0,
        alignment_error_normalized_by_interocular=0.04,
        landmark_geometry_valid=True,
        finite_embedding=True,
        composite_quality_score=0.8,
    )
    assert q.hard_rejection_reasons == []


def test_face_observation_rejects_bad_landmark_order() -> None:
    with pytest.raises(ValidationError):
        FaceObservation(
            observation_id="obs_1",
            source_id="Friends.mp4",
            frame_index=0,
            pts=0,
            time_base_num=1,
            time_base_den=30,
            pts_ns=0,
            frame_width=1920,
            frame_height=1080,
            detection_ordinal=0,
            bbox_xyxy=BBoxXYXY(x1=100.0, y1=100.0, x2=200.0, y2=300.0),
            detector_score=0.9,
            landmarks_5=Landmarks5(
                left_eye=(1.0, 2.0),
                right_eye=(3.0, 4.0),
                nose=(5.0, 6.0),
                left_mouth=(7.0, 8.0),
                right_mouth=(9.0, 10.0),
            ),
            landmark_order=["left_eye", "right_eye", "nose", "left_mouth"],
            quality=QualityMetrics(
                bbox_width_px=100.0,
                bbox_height_px=120.0,
                bbox_min_side_px=100.0,
                bbox_area_px=12000.0,
                detector_score=0.9,
                grayscale_laplacian_variance=100.0,
                brightness_mean=128.0,
                brightness_std=30.0,
                dark_clip_fraction=0.0,
                bright_clip_fraction=0.0,
                interocular_distance_px=50.0,
                alignment_reprojection_error_px=2.0,
                alignment_error_normalized_by_interocular=0.04,
                landmark_geometry_valid=True,
                finite_embedding=True,
                composite_quality_score=0.8,
            ),
            tracking_eligible=True,
            recognition_eligible=True,
        )


def test_ground_truth_schema_version() -> None:
    gt = GroundTruth(video_sha256="abc")
    assert gt.schema_version == "mv-video-ground-truth/v1"
    assert gt.anchors == []
