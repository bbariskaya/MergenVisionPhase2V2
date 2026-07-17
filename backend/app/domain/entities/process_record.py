"""Process record entity tracking recognition/enrollment/delete flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.domain.errors import InvalidTransitionError
from app.domain.value_objects import ProcessId


@dataclass
class ProcessRecord:
    process_id: ProcessId
    process_type: str
    status: str = "processing"  # "processing" | "completed" | "failed"
    face_count: int | None = None
    error_code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    cancelled_at: datetime | None = None

    def set_details(self, details: dict[str, Any]) -> None:
        self.details = details or {}

    def complete(self, face_count: int, details: dict[str, Any] | None = None) -> None:
        if self.status != "processing":
            raise InvalidTransitionError(f"Cannot complete process with status {self.status}")
        self.status = "completed"
        self.face_count = face_count
        self.error_code = None
        self.failed_at = None
        self.cancelled_at = None
        self.details = details or {}
        self.completed_at = datetime.now(UTC)

    def fail(self, error_code: str, details: dict[str, Any] | None = None) -> None:
        if self.status != "processing":
            raise InvalidTransitionError(f"Cannot fail process with status {self.status}")
        now = datetime.now(UTC)
        self.status = "failed"
        self.error_code = error_code
        self.face_count = None
        self.cancelled_at = None
        self.details = details or {}
        self.completed_at = now
        self.failed_at = now

    def cancel(self, details: dict[str, Any] | None = None) -> None:
        if self.status != "processing":
            raise InvalidTransitionError(f"Cannot cancel process with status {self.status}")
        now = datetime.now(UTC)
        self.status = "cancelled"
        self.error_code = None
        self.face_count = None
        self.failed_at = None
        self.details = details or {}
        self.completed_at = now
        self.cancelled_at = now
