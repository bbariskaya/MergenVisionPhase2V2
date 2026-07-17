"""Common device-resident face pipeline port.

Milestone 4 exposes a single FacePipeline for both image (nvJPEG -> GPU surface)
and video (NVDEC/NVMM -> GPU surface) paths. This module defines the Python
contract only; the native implementation lives under `backend/native/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.domain.value_objects import BoundingBox


@dataclass(frozen=True)
class DeviceImageView:
    """Lightweight descriptor for a device-resident image surface.

    The pipeline does **not** own the underlying memory unless
    ``ownership == "pipeline"``. Callers must keep the surface alive for the
    lifetime of ``infer_device_batch``.
    """

    device_pointer: int
    width: int
    height: int
    pitch: int
    pixel_format: str
    device_id: int = 0
    source_batch_index: int = 0
    frame_index: int = 0
    pts_ns: int = 0
    display_width: int | None = None
    display_height: int | None = None
    rotation: int = 0
    ownership: str = "external"


@dataclass(frozen=True)
class FaceObservation:
    """One face observation emitted by the common device pipeline."""

    detection_index: int
    ordinal: int
    bbox: BoundingBox
    landmarks5: tuple[float, ...]
    detector_score: float
    quality_score: float
    tracking_eligible: bool
    recognition_eligible: bool
    rejection_code: str = ""
    embedding: tuple[float, ...] = field(default_factory=tuple)
    model_version: str = ""
    preprocess_version: str = ""


@dataclass(frozen=True)
class FaceObservations:
    """Observations for a single input view, preserving frame association."""

    source_batch_index: int
    frame_index: int
    pts_ns: int
    display_width: int
    display_height: int
    detections: tuple[FaceObservation, ...]


class FacePipeline(Protocol):
    """Common device-resident detector/recognizer pipeline port.

    Implementations must:

    - not free external NVMM surfaces
    - preserve frame association across partial/final batches
    - return original-resolution bounding boxes
    - return finite, L2-normalized 512-D embeddings when recognition eligible
    - never perform CPU-hosted full-frame decode
    """

    async def infer_device_batch(
        self,
        views: tuple[DeviceImageView, ...],
    ) -> tuple[FaceObservations, ...]:
        """Run detection + alignment + embedding on a batch of device surfaces."""
        ...
