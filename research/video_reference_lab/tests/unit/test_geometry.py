"""Unit tests for geometry helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mergenvision_video_lab.geometry import (
    bbox_area,
    clamp_bbox,
    interocular_distance_px,
    iou_xyxy,
)


def test_clamp_bbox() -> None:
    x1, y1, x2, y2 = clamp_bbox(-10.0, -5.0, 50.0, 60.0, 40.0, 50.0)
    assert (x1, y1, x2, y2) == (0.0, 0.0, 40.0, 50.0)


def test_clamp_bbox_degenerate() -> None:
    x1, y1, x2, y2 = clamp_bbox(10.0, 20.0, 10.0, 30.0, 100.0, 100.0)
    assert x1 == x2 == 10.0


def test_bbox_area() -> None:
    assert bbox_area(0.0, 0.0, 10.0, 20.0) == 200.0
    assert bbox_area(0.0, 0.0, 0.0, 10.0) == 0.0


def test_iou_identical() -> None:
    box = [0.0, 0.0, 10.0, 10.0]
    assert iou_xyxy(box, box) == 1.0


def test_iou_no_overlap() -> None:
    a = [0.0, 0.0, 10.0, 10.0]
    b = [20.0, 20.0, 30.0, 30.0]
    assert iou_xyxy(a, b) == 0.0


def test_iou_partial() -> None:
    a = [0.0, 0.0, 10.0, 10.0]
    b = [5.0, 0.0, 15.0, 10.0]
    iou = iou_xyxy(a, b)
    assert 0.0 < iou < 1.0
    assert iou == pytest.approx(1 / 3, abs=0.01)


def test_interocular_distance() -> None:
    landmarks = np.array(
        [[0.0, 0.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    assert interocular_distance_px(landmarks) == 5.0


def test_interocular_distance_bad_shape() -> None:
    with pytest.raises(ValueError):
        interocular_distance_px(np.array([[0.0, 0.0], [1.0, 1.0]]))
