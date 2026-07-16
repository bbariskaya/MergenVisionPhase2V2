"""Unit tests for ArcFace five-point alignment."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mergenvision_video_lab.alignment import (
    ARCFACE_TEMPLATE_112,
    AlignmentError,
    align_face,
)


def _synthetic_face_image(size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic gradient image and plausible landmarks."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Draw a simple face-like pattern.
    y, x = np.mgrid[0:size, 0:size]
    mask = ((x - size // 2) ** 2 + (y - size // 2) ** 2) < (size // 3) ** 2
    img[mask] = [80, 120, 160]

    center = np.array([size / 2, size / 2], dtype=np.float32)
    landmarks = np.array(
        [
            center + [-30, -20],  # left eye
            center + [30, -20],  # right eye
            center + [0, 10],  # nose
            center + [-20, 40],  # left mouth
            center + [20, 40],  # right mouth
        ],
        dtype=np.float32,
    )
    return img, landmarks


def test_alignment_output_shape_and_type() -> None:
    img, landmarks = _synthetic_face_image()
    crop, matrix, error = align_face(img, landmarks, output_size=112)
    assert crop.shape == (112, 112, 3)
    assert crop.dtype == np.uint8
    assert matrix.shape == (2, 3)
    assert error >= 0.0


def test_alignment_parity_with_insightface_norm_crop() -> None:
    from insightface.utils.face_align import norm_crop

    img, landmarks = _synthetic_face_image()
    my_crop, _, _ = align_face(img, landmarks, output_size=112, color_order="BGR")
    oracle_crop = norm_crop(img, landmarks, image_size=112, mode="arcface")

    diff = np.abs(my_crop.astype(np.float32) - oracle_crop.astype(np.float32))
    mean_abs_error = float(np.mean(diff))
    max_error = float(np.max(diff))
    assert mean_abs_error < 2.0, f"MAE vs InsightFace norm_crop: {mean_abs_error}"
    assert max_error < 20.0, f"max error vs InsightFace norm_crop: {max_error}"


def test_eye_swap_changes_output() -> None:
    img, landmarks = _synthetic_face_image()
    correct_crop, _, _ = align_face(img, landmarks)
    swapped = landmarks.copy()
    swapped[0], swapped[1] = landmarks[1].copy(), landmarks[0].copy()
    swapped_crop, _, _ = align_face(img, swapped)
    diff = np.mean(np.abs(correct_crop.astype(np.float32) - swapped_crop.astype(np.float32)))
    assert diff > 10.0, "eye-swap should produce visibly different crop"


def test_rgb_input_matches_bgr_after_color_conversion() -> None:
    img, landmarks = _synthetic_face_image()
    bgr_crop, _, _ = align_face(img, landmarks, color_order="BGR")
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_crop, _, _ = align_face(rgb_img, landmarks, color_order="RGB")
    # Same scene, different documented input order -> RGB output equals
    # BGR output converted to RGB.
    np.testing.assert_allclose(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB), rgb_crop, atol=1)


def test_degenerate_landmarks_rejected() -> None:
    img, _ = _synthetic_face_image()
    bad = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)
    with pytest.raises(AlignmentError):
        align_face(img, bad)


def test_nan_landmarks_rejected() -> None:
    img, _ = _synthetic_face_image()
    bad = np.full((5, 2), np.nan, dtype=np.float32)
    with pytest.raises(AlignmentError):
        align_face(img, bad)


def test_arcface_template_has_expected_order() -> None:
    # Left eye x < right eye x in the canonical template.
    assert ARCFACE_TEMPLATE_112[0, 0] < ARCFACE_TEMPLATE_112[1, 0]
    assert ARCFACE_TEMPLATE_112.shape == (5, 2)
