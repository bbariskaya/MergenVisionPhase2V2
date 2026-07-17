"""Port for native image face detection + embedding inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.domain.value_objects import BoundingBox


@dataclass(frozen=True)
class NativeFaceDetection:
    bounding_box: BoundingBox
    detector_confidence: float
    embedding: list[float]
    aligned_crop_bytes: bytes


@dataclass(frozen=True)
class NativeRecognitionResult:
    image_width: int
    image_height: int
    detections: list[NativeFaceDetection]


class ImageRecognitionEngine(ABC):
    """Abstract interface to the native GPU image recognition runtime."""

    @abstractmethod
    async def detect_and_embed(self, image_bytes: bytes) -> NativeRecognitionResult:
        """Run detector + recognizer on a JPEG/PNG/WebP byte stream."""
        ...
