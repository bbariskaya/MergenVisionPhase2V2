"""Python wrapper around the Phase 1 native CUDA operator extension.

The compiled extension is ``_mv_phase1_bulk_native``.  This module re-exports
its functions with the same names used by the reference MergenVisionDemo
implementation so the adapted Python code requires minimal changes.
"""

from __future__ import annotations

from mv_phase1_bulk._mv_phase1_bulk_native import (
    argsort_descending,
    l2_normalize,
    nchw_float_to_hwc_uint8,
    nms,
    retinaface_decode_batch,
    retinaface_pick_largest,
    scale_clip_compact,
    scale_clip_compact_xy,
    similarity_transform,
    warp_align,
)

__all__ = [
    "argsort_descending",
    "l2_normalize",
    "nchw_float_to_hwc_uint8",
    "nms",
    "retinaface_decode_batch",
    "retinaface_pick_largest",
    "scale_clip_compact",
    "scale_clip_compact_xy",
    "similarity_transform",
    "warp_align",
]
