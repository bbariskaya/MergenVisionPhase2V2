"""Face identity aggregate root."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.domain.errors import InvalidTransitionError, ValidationError
from app.domain.value_objects import FaceId


@dataclass
class FaceIdentity:
    face_id: FaceId
    status: str = "anonymous"  # "anonymous" | "known"
    is_active: bool = True
    display_name: str | None = None
    identity_metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.status not in {"anonymous", "known"}:
            raise ValidationError(f"Invalid status: {self.status}")
        if self.status == "known" and (not self.display_name or not self.display_name.strip()):
            raise ValidationError("known identity requires a non-empty display_name")

    def promote_to_known(self, display_name: str, metadata: dict[str, Any] | None = None) -> None:
        if not self.is_active:
            raise InvalidTransitionError("Cannot promote inactive identity")
        if self.status != "anonymous":
            raise InvalidTransitionError(f"Cannot promote identity with status {self.status}")
        if not display_name or not display_name.strip():
            raise ValidationError("display_name is required for known identity")
        self.display_name = display_name.strip()
        self.identity_metadata = metadata or {}
        self.status = "known"
        self.version += 1
        self.updated_at = datetime.now(UTC)

    def deactivate(self) -> None:
        if not self.is_active:
            return
        self.is_active = False
        self.deleted_at = datetime.now(UTC)
        self.updated_at = self.deleted_at
