"""Face image quality metrics and composite score."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from mergenvision_video_lab.contracts import QualityMetrics
from mergenvision_video_lab.geometry import interocular_distance_px


def compute_laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian on a grayscale uint8 image (blur metric)."""
    if gray.dtype != np.uint8:
        raise ValueError("laplacian variance expects uint8 grayscale")
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))


def compute_brightness_stats(gray: np.ndarray) -> tuple[float, float]:
    """Return (mean, std) of grayscale uint8 image."""
    return float(np.mean(gray)), float(np.std(gray))


def compute_clip_fractions(gray: np.ndarray) -> tuple[float, float]:
    """Return (dark_clip_fraction, bright_clip_fraction) for [0,255] uint8."""
    total = gray.size
    dark = int(np.sum(gray <= 0))
    bright = int(np.sum(gray >= 255))
    return dark / total, bright / total


def compute_quality(
    aligned_crop: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    landmarks_5: np.ndarray,
    detector_score: float,
    reprojection_error_px: float,
    config: dict[str, Any],
) -> QualityMetrics:
    """Compute quality metrics and rejection reasons for one observation."""
    x1, y1, x2, y2 = bbox_xyxy
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_min_side = min(bbox_width, bbox_height)
    bbox_area = bbox_width * bbox_height

    gray = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY)
    lap_var = compute_laplacian_variance(gray)
    brightness_mean, brightness_std = compute_brightness_stats(gray)
    dark_clip, bright_clip = compute_clip_fractions(gray)

    iod = interocular_distance_px(landmarks_5)
    alignment_error_normalized = reprojection_error_px / iod if iod > 1e-6 else float("inf")

    reasons: list[str] = []
    if bbox_min_side < config["min_face_side_px"]:
        reasons.append("face_too_small")
    if detector_score < config["min_detector_score_recognition"]:
        reasons.append("detector_score_low")
    if lap_var < config["min_laplacian_variance"]:
        reasons.append("too_blurry")
    if alignment_error_normalized > config["max_alignment_error_normalized"]:
        reasons.append("alignment_residual_high")
    if brightness_mean < config["min_brightness_mean"]:
        reasons.append("too_dark")
    if brightness_mean > config["max_brightness_mean"]:
        reasons.append("too_bright")
    if dark_clip > config["max_dark_clip_fraction"]:
        reasons.append("dark_clipped")
    if bright_clip > config["max_bright_clip_fraction"]:
        reasons.append("bright_clipped")

    # Component normalization formulas (explicit, clamped to [0,1]).
    detector_component = float(np.clip(detector_score, 0.0, 1.0))

    frame_min_side = min(frame_width, frame_height)
    size_component = float(np.clip(bbox_min_side / max(frame_min_side, 1.0), 0.0, 1.0))

    # Log-scaled laplacian variance; reference maximum 1000.
    lap_ref = 1000.0
    lap_component = float(np.clip(np.log1p(lap_var) / np.log1p(lap_ref), 0.0, 1.0))

    # Exposure centered at 128.
    exposure_component = float(np.clip(1.0 - abs(brightness_mean - 128.0) / 128.0, 0.0, 1.0))

    # Alignment error; reference maximum 0.20 normalized.
    align_max = 0.20
    align_component = float(np.clip(1.0 - alignment_error_normalized / align_max, 0.0, 1.0))

    composite = (
        0.30 * detector_component
        + 0.25 * size_component
        + 0.20 * lap_component
        + 0.15 * exposure_component
        + 0.10 * align_component
    )

    return QualityMetrics(
        bbox_width_px=bbox_width,
        bbox_height_px=bbox_height,
        bbox_min_side_px=bbox_min_side,
        bbox_area_px=bbox_area,
        detector_score=detector_score,
        grayscale_laplacian_variance=lap_var,
        brightness_mean=brightness_mean,
        brightness_std=brightness_std,
        dark_clip_fraction=dark_clip,
        bright_clip_fraction=bright_clip,
        interocular_distance_px=iod,
        alignment_reprojection_error_px=reprojection_error_px,
        alignment_error_normalized_by_interocular=alignment_error_normalized,
        landmark_geometry_valid=True,
        finite_embedding=False,  # set by caller after embedding
        composite_quality_score=composite,
        hard_rejection_reasons=reasons,
    )
