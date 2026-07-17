"""Native GPU image recognition adapter.

This adapter owns the `image_runtime` pybind11 extension and translates its
compact output into domain objects. It is infrastructure: it knows nothing about
faceId, known/anonymous semantics, or business rules.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.application.ports.image_recognition import (
    ImageRecognitionEngine,
    NativeFaceDetection,
    NativeRecognitionResult,
)
from app.domain.errors import DomainError
from app.domain.value_objects import BoundingBox
from app.infrastructure.config import settings
from app.infrastructure.model_profile import ModelProfile

logger = logging.getLogger(__name__)


class NativeRecognitionError(DomainError):
    """Raised when the native runtime fails to process an image."""


class NativeImageRecognitionAdapter(ImageRecognitionEngine):
    def __init__(
        self,
        model_profile_path: str | None = None,
        detector_engine_path: str | None = None,
        recognizer_engine_path: str | None = None,
        gpu_device_id: int | None = None,
    ) -> None:
        import image_runtime

        self._model_profile_path = model_profile_path or settings.model_profile_path
        self._detector_engine_path = detector_engine_path or settings.detector_engine_path
        self._recognizer_engine_path = recognizer_engine_path or settings.recognizer_engine_path
        self._gpu_device_id = gpu_device_id if gpu_device_id is not None else settings.gpu_device_id

        try:
            profile = ModelProfile.load(self._model_profile_path)
            self._runtime = image_runtime.ImageRuntime(
                profile.model_dump(by_alias=True),
                self._detector_engine_path,
                self._recognizer_engine_path,
                self._gpu_device_id,
                settings.inference_slot_count,
            )
        except Exception as exc:
            raise NativeRecognitionError(f"Failed to initialize native image runtime: {exc}") from exc

    async def detect_and_embed(self, image_bytes: bytes) -> NativeRecognitionResult:
        loop = asyncio.get_running_loop()
        try:
            raw = await loop.run_in_executor(None, self._runtime.infer_jpeg, image_bytes)
        except Exception as exc:
            raise NativeRecognitionError(f"Native inference failed: {exc}") from exc

        return self._to_domain(raw)

    def _to_domain(self, raw: dict[str, Any]) -> NativeRecognitionResult:
        detections: list[NativeFaceDetection] = []
        for det in raw.get("detections", []):
            bbox = det["bbox"]
            detections.append(
                NativeFaceDetection(
                    bounding_box=BoundingBox(
                        x=int(round(bbox["x"])),
                        y=int(round(bbox["y"])),
                        width=int(round(bbox["width"])),
                        height=int(round(bbox["height"])),
                    ),
                    detector_confidence=float(det["detector_confidence"]),
                    embedding=list(det["embedding"]),
                    aligned_crop_bytes=bytes(det["aligned_crop_bytes"]),
                )
            )
        return NativeRecognitionResult(
            image_width=int(raw["image_width"]),
            image_height=int(raw["image_height"]),
            detections=detections,
        )
