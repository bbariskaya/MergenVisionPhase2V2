"""Face sample entity representing one biometric capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.domain.errors import InvalidTransitionError
from app.domain.value_objects import FaceId, SampleId


@dataclass
class FaceSample:
    sample_id: SampleId
    face_id: FaceId
    state: str = "pending"  # "pending" | "active" | "failed" | "inactive"
    bucket: str | None = None
    object_key: str | None = None
    failure_code: str | None = None
    is_active: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    activated_at: datetime | None = None
    deactivated_at: datetime | None = None

    def mark_active(self, bucket: str, object_key: str) -> None:
        if self.state != "pending":
            raise InvalidTransitionError(f"Cannot mark active from state {self.state}")
        self.state = "active"
        self.bucket = bucket
        self.object_key = object_key
        self.is_active = True
        self.activated_at = datetime.now(UTC)

    def mark_failed(self, failure_code: str) -> None:
        if self.state not in {"pending", "active"}:
            raise InvalidTransitionError(f"Cannot mark failed from state {self.state}")
        self.state = "failed"
        self.failure_code = failure_code
        self.is_active = False

    def mark_inactive(self) -> None:
        if self.state != "active":
            raise InvalidTransitionError(f"Cannot mark inactive from state {self.state}")
        self.state = "inactive"
        self.is_active = False
        self.deactivated_at = datetime.now(UTC)
