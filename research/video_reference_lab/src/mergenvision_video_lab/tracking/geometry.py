"""Bounding-box geometry helpers adapted for the ByteTrack xyah Kalman model."""

from __future__ import annotations

import numpy as np


def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
    """Convert top-left-width-height to center-x, center-y, aspect-ratio, height."""
    ret = tlwh.copy()
    ret[:2] += ret[2:] / 2.0
    ret[2] /= ret[3]
    return ret


def xyah_to_tlwh(xyah: np.ndarray) -> np.ndarray:
    """Convert center-x, center-y, aspect-ratio, height to tlwh."""
    ret = xyah.copy()
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2.0
    return ret


def tlwh_to_tlbr(tlwh: np.ndarray) -> np.ndarray:
    """Convert tlwh to xyxy (top-left, bottom-right)."""
    ret = tlwh.copy()
    ret[2:] += ret[:2]
    return ret


def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
    """Convert xyxy to tlwh."""
    ret = tlbr.copy()
    ret[2:] -= ret[:2]
    return ret


def bbox_xyxy_to_xyah(xyxy: np.ndarray) -> np.ndarray:
    """Convert XYXY to xyah."""
    return tlwh_to_xyah(tlbr_to_tlwh(xyxy))


def bbox_xyxy_to_tlwh(xyxy: np.ndarray) -> np.ndarray:
    """Convert XYXY to tlwh."""
    return tlbr_to_tlwh(xyxy)
