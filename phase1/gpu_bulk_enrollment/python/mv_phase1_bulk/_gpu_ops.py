"""Python wrapper around the Phase 1 native CUDA operator extension.

The compiled extension is ``_mv_phase1_bulk_native``.  This module re-exports
its functions with the same names used by the reference MergenVisionDemo
implementation so the adapted Python code requires minimal changes.
"""
from __future__ import annotations

from mv_phase1_bulk import _mv_phase1_bulk_native as _native

l2_normalize = _native.l2_normalize
similarity_transform = _native.similarity_transform
nms = _native.nms
scale_clip_compact = _native.scale_clip_compact
scale_clip_compact_xy = _native.scale_clip_compact_xy
scrfd_decode_level = _native.scrfd_decode_level
retinaface_decode_batch = _native.retinaface_decode_batch
retinaface_pick_largest = _native.retinaface_pick_largest
argsort_descending = _native.argsort_descending
warp_align = _native.warp_align
spin_wait_cycles = _native.spin_wait_cycles
