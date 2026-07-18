"""Person aggregate root.

A Person is the human-facing identity that owns one or more FaceIdentity
biometric records.  Names and PII-style metadata live here, never in Qdrant or
MinIO keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.domain.errors import InvalidTransitionError, ValidationError
from app.domain.value_objects import PersonId


@dataclass
class Person:
    person_id: PersonId
    display_name: str
    person_metadata: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.display_name or not self.display_name.strip():
            raise ValidationError("Person display_name is required")

    @classmethod
    def create(
        cls,
        display_name: str,
        person_metadata: dict[str, Any] | None = None,
    ) -> Person:
        from uuid import uuid4

        return cls(
            person_id=PersonId(uuid4()),
            display_name=display_name,
            person_metadata=person_metadata or {},
        )

    def rename(self, display_name: str, metadata: dict[str, Any] | None = None) -> None:
        if not self.is_active:
            raise InvalidTransitionError("Cannot rename inactive person")
        if not display_name or not display_name.strip():
            raise ValidationError("Person display_name is required")
        self.display_name = display_name.strip()
        if metadata is not None:
            self.person_metadata = metadata
        self.version += 1
        self.updated_at = datetime.now(UTC)

    def update_metadata(self, metadata: dict[str, Any]) -> None:
        if not self.is_active:
            raise InvalidTransitionError("Cannot update metadata for inactive person")
        self.person_metadata = metadata
        self.version += 1
        self.updated_at = datetime.now(UTC)

    def deactivate(self) -> None:
        if not self.is_active:
            return
        self.is_active = False
        self.deleted_at = datetime.now(UTC)
        self.updated_at = self.deleted_at
        self.version += 1
