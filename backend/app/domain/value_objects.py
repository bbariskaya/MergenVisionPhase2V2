"""Immutable value objects shared across the domain."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import NewType

FaceId = NewType("FaceId", uuid.UUID)
PersonId = NewType("PersonId", uuid.UUID)
SampleId = NewType("SampleId", uuid.UUID)
ProcessId = NewType("ProcessId", uuid.UUID)
ResultId = NewType("ResultId", uuid.UUID)
VideoId = NewType("VideoId", uuid.UUID)
JobId = NewType("JobId", uuid.UUID)
UploadSessionId = NewType("UploadSessionId", uuid.UUID)


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("BoundingBox x and y must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("BoundingBox width and height must be positive")


@dataclass(frozen=True)
class ObjectStat:
    bucket: str
    key: str
    size: int
    content_type: str | None = None
    sha256: str | None = None
