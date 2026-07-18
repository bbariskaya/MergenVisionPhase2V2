"""Face identity aggregate root."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.domain.errors import InvalidTransitionError, ValidationError
from app.domain.value_objects import FaceId, PersonId


@dataclass
class FaceIdentity:
    face_id: FaceId
    status: str = "anonymous"  # "anonymous" | "known"
    is_active: bool = True
    display_name: str | None = None
    identity_metadata: dict[str, Any] = field(default_factory=dict)
    person_id: PersonId | None = None
    redirect_to_face_id: FaceId | None = None
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    deleted_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.status not in {"anonymous", "known"}:
            raise ValidationError(f"Invalid status: {self.status}")
        if self.status == "known":
            if self.person_id is None:
                raise ValidationError("known identity requires a person_id")
            if not self.display_name or not self.display_name.strip():
                raise ValidationError("known identity requires a non-empty display_name")
        if self.redirect_to_face_id is not None and self.is_active:
            raise ValidationError("redirected identity must be inactive")

    @property
    def canonical_face_id(self) -> FaceId:
        """Return the canonical face_id after following any redirect."""
        return self.redirect_to_face_id if self.redirect_to_face_id is not None else self.face_id

    def promote_to_known(
        self,
        person_id: PersonId,
        display_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.is_active:
            raise InvalidTransitionError("Cannot promote inactive identity")
        if self.redirect_to_face_id is not None:
            raise InvalidTransitionError("Cannot promote a redirected identity")
        if self.status != "anonymous":
            raise InvalidTransitionError(f"Cannot promote identity with status {self.status}")
        if person_id is None:
            raise ValidationError("person_id is required for known identity")
        if not display_name or not display_name.strip():
            raise ValidationError("display_name is required for known identity")
        self.person_id = person_id
        self.display_name = display_name.strip()
        self.identity_metadata = metadata or {}
        self.status = "known"
        self.version += 1
        self.updated_at = datetime.now(UTC)

    def assign_to_person(self, person_id: PersonId, canonical_face_id: FaceId) -> None:
        """Merge this identity into an existing person.

        The face_identity becomes inactive and redirects to the canonical
        face_id. Historical snapshots remain untouched.
        """
        if not self.is_active:
            raise InvalidTransitionError("Cannot assign inactive identity")
        if self.redirect_to_face_id is not None:
            raise InvalidTransitionError("Identity is already redirected")
        if canonical_face_id == self.face_id:
            raise ValidationError("Cannot redirect an identity to itself")
        self.person_id = None
        self.redirect_to_face_id = canonical_face_id
        self.is_active = False
        self.deleted_at = datetime.now(UTC)
        self.display_name = None
        self.identity_metadata = {}
        self.version += 1
        self.updated_at = self.deleted_at

    def deactivate(self) -> None:
        if not self.is_active:
            return
        self.is_active = False
        self.deleted_at = datetime.now(UTC)
        self.updated_at = self.deleted_at
        self.version += 1
