"""Association cost functions and Hungarian assignment.

Adapted from FoundationVision/ByteTrack
(https://github.com/FoundationVision/ByteTrack), yolox/tracker/matching.py,
commit d1bf0191adff59bc8fcfeaa0b33d3d1642552a99, MIT license.

Local changes:
- Replaced ``lap.lapjv`` with ``scipy.optimize.linear_sum_assignment``.
- Replaced Cython ``bbox_ious`` with the project's own IoU helper.
- Removed embedding_distance / fuse_motion helpers not used by the face lab.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from mergenvision_video_lab.geometry import iou_xyxy


_GATED_COST = 1e5


def iou_distance(atracks: Sequence[np.ndarray], btracks: Sequence[np.ndarray]) -> np.ndarray:
    """Compute 1 - IoU cost matrix from arrays of tlbr boxes."""
    n, m = len(atracks), len(btracks)
    cost = np.zeros((n, m), dtype=np.float64)
    if n == 0 or m == 0:
        return cost
    for i, a in enumerate(atracks):
        for j, b in enumerate(btracks):
            cost[i, j] = 1.0 - iou_xyxy(a, b)
    return cost


def fuse_score(cost_matrix: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Fuse IoU distance with detection scores (ByteTrack ``fuse_score``)."""
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1.0 - cost_matrix
    det_scores = np.asarray(scores).reshape(1, -1)
    det_scores = np.repeat(det_scores, cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1.0 - fuse_sim


def gated_iou_distance(
    track_boxes: Sequence[np.ndarray],
    det_boxes: Sequence[np.ndarray],
    min_iou: float,
) -> np.ndarray:
    """IoU distance with hard gating: pairs below ``min_iou`` get a large cost."""
    cost = iou_distance(track_boxes, det_boxes)
    if cost.size == 0:
        return cost
    ious = 1.0 - cost
    cost[ious < min_iou] = _GATED_COST
    return cost


def linear_assignment(
    cost_matrix: np.ndarray,
    thresh: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment with a cost threshold.

    Returns ``(matches, unmatched_rows, unmatched_cols)``.  Any matched pair
    whose cost exceeds ``thresh`` is treated as unmatched.
    """
    if cost_matrix.size == 0:
        rows = list(range(cost_matrix.shape[0]))
        cols = list(range(cost_matrix.shape[1]))
        return [], rows, cols

    rows, cols = linear_sum_assignment(cost_matrix)
    matches: list[tuple[int, int]] = []
    unmatched_rows: list[int] = []
    unmatched_cols: list[int] = []

    matched_rows = set()
    matched_cols = set()
    for r, c in zip(rows, cols):
        if cost_matrix[r, c] <= thresh:
            matches.append((int(r), int(c)))
            matched_rows.add(r)
            matched_cols.add(c)

    for r in range(cost_matrix.shape[0]):
        if r not in matched_rows:
            unmatched_rows.append(r)
    for c in range(cost_matrix.shape[1]):
        if c not in matched_cols:
            unmatched_cols.append(c)

    return matches, unmatched_rows, unmatched_cols
