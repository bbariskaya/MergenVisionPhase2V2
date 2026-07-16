"""Immutable recognition result snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.domain.errors import ValidationError
from app.domain.value_objects import BoundingBox, FaceId, ProcessId, ResultId, SampleId


@dataclass(frozen=True)
class RecognitionResult:
    result_id: ResultId
    process_id: ProcessId
    face_id: FaceId
    status: str  # "known" | "anonymous" | "new_anonymous"
    bounding_box: BoundingBox
    match_confidence: float
    sample_id: SampleId | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"known", "anonymous", "new_anonymous"}:
            raise ValidationError(f"Invalid recognition result status: {self.status}")
        if not (0.0 <= self.match_confidence <= 1.0):
            raise ValidationError("match_confidence must be between 0.0 and 1.0")
