"""Common device-resident face pipeline adapter (M4/M5 scaffold).

This adapter currently exposes the Python contract. The real native
implementation must be built inside the pinned GPU/DeepStream container using
``backend/native/face_pipeline/`` and then bound here.

Until then ``infer_device_batch`` raises ``VideoExecutorNotReadyError`` so that
production readiness reporting is honest and no fake/placeholder pipeline is
presented as working.
"""

from __future__ import annotations

from app.application.ports.face_pipeline import (
    DeviceImageView,
    FaceObservations,
    FacePipeline,
)
from app.domain.errors import VideoExecutorNotReadyError


class FacePipelineAdapter(FacePipeline):
    """Host-side placeholder for the common device FacePipeline."""

    async def infer_device_batch(
        self,
        views: tuple[DeviceImageView, ...],
    ) -> tuple[FaceObservations, ...]:
        del views  # not yet implemented
        raise VideoExecutorNotReadyError(
            "Common device FacePipeline is not available in this runtime. "
            "Run inside the pinned DeepStream/GPU container."
        )
