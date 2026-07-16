"""Unit tests for quality metrics."""

from __future__ import annotations

import numpy as np

from mergenvision_video_lab.alignment import align_face
from mergenvision_video_lab.quality import compute_quality


def _synthetic_face() -> tuple[np.ndarray, np.ndarray]:
    size = 256
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = [128, 128, 128]
    center = np.array([size / 2, size / 2], dtype=np.float32)
    landmarks = np.array(
        [
            center + [-30, -20],
            center + [30, -20],
            center + [0, 10],
            center + [-20, 40],
            center + [20, 40],
        ],
        dtype=np.float32,
    )
    crop, _, error = align_face(img, landmarks)
    return crop, landmarks, error


def test_quality_metrics_finite() -> None:
    crop, landmarks, error = _synthetic_face()
    q = compute_quality(
        aligned_crop=crop,
        bbox_xyxy=(50.0, 50.0, 150.0, 150.0),
        frame_width=256,
        frame_height=256,
        landmarks_5=landmarks,
        detector_score=0.95,
        reprojection_error_px=error,
        config={
            "min_face_side_px": 32,
            "min_detector_score_recognition": 0.60,
            "min_laplacian_variance": 40.0,
            "max_alignment_error_normalized": 0.10,
            "min_brightness_mean": 30.0,
            "max_brightness_mean": 225.0,
            "max_dark_clip_fraction": 0.40,
            "max_bright_clip_fraction": 0.40,
        },
    )
    assert q.bbox_min_side_px == 100.0
    assert q.detector_score == 0.95
    assert np.isfinite(q.composite_quality_score)
    assert 0.0 <= q.composite_quality_score <= 1.0


def test_quality_rejects_too_small() -> None:
    crop, landmarks, error = _synthetic_face()
    q = compute_quality(
        aligned_crop=crop,
        bbox_xyxy=(50.0, 50.0, 60.0, 60.0),
        frame_width=256,
        frame_height=256,
        landmarks_5=landmarks,
        detector_score=0.95,
        reprojection_error_px=error,
        config={
            "min_face_side_px": 32,
            "min_detector_score_recognition": 0.60,
            "min_laplacian_variance": 40.0,
            "max_alignment_error_normalized": 0.10,
            "min_brightness_mean": 30.0,
            "max_brightness_mean": 225.0,
            "max_dark_clip_fraction": 0.40,
            "max_bright_clip_fraction": 0.40,
        },
    )
    assert "face_too_small" in q.hard_rejection_reasons


def test_quality_determinism() -> None:
    crop, landmarks, error = _synthetic_face()
    config = {
        "min_face_side_px": 32,
        "min_detector_score_recognition": 0.60,
        "min_laplacian_variance": 40.0,
        "max_alignment_error_normalized": 0.10,
        "min_brightness_mean": 30.0,
        "max_brightness_mean": 225.0,
        "max_dark_clip_fraction": 0.40,
        "max_bright_clip_fraction": 0.40,
    }
    q1 = compute_quality(crop, (50.0, 50.0, 150.0, 150.0), 256, 256, landmarks, 0.9, error, config)
    q2 = compute_quality(crop, (50.0, 50.0, 150.0, 150.0), 256, 256, landmarks, 0.9, error, config)
    assert q1.composite_quality_score == q2.composite_quality_score
