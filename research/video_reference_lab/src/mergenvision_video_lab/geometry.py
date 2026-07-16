"""Geometry helpers for bounding boxes and landmarks."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def clamp_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    """Clamp and order an XYXY bbox to [0,width] x [0,height].

    Returns (x1, y1, x2, y2) with x2 > x1 and y2 > y1. Degenerate boxes are
    not repaired beyond clamping; callers must reject zero-area results.
    """
    x1 = float(np.clip(x1, 0.0, width))
    y1 = float(np.clip(y1, 0.0, height))
    x2 = float(np.clip(x2, 0.0, width))
    y2 = float(np.clip(y2, 0.0, height))
    return x1, y1, x2, y2


def bbox_area(x1: float, y1: float, x2: float, y2: float) -> float:
    """Area of an XYXY bbox."""
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    """Intersection-over-union for two XYXY boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def interocular_distance_px(landmarks_5: np.ndarray) -> float:
    """Euclidean distance between left and right eye in pixels."""
    if landmarks_5.shape != (5, 2):
        raise ValueError("landmarks_5 must have shape (5, 2)")
    left_eye = landmarks_5[0]
    right_eye = landmarks_5[1]
    return float(np.linalg.norm(left_eye - right_eye))
