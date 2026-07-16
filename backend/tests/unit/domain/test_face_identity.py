"""Unit tests for FaceIdentity state transitions."""

import uuid

import pytest

from app.domain.entities.face_identity import FaceIdentity
from app.domain.errors import InvalidTransitionError, ValidationError
from app.domain.value_objects import FaceId


def _face_id() -> FaceId:
    return FaceId(uuid.uuid4())


def test_new_identity_defaults_to_anonymous() -> None:
    identity = FaceIdentity(face_id=_face_id())
    assert identity.status == "anonymous"
    assert identity.is_active is True
    assert identity.display_name is None


def test_promote_anonymous_to_known() -> None:
    identity = FaceIdentity(face_id=_face_id())
    identity.promote_to_known("Alice", {"department": "Engineering"})
    assert identity.status == "known"
    assert identity.display_name == "Alice"
    assert identity.identity_metadata == {"department": "Engineering"}
    assert identity.version == 2


def test_promote_requires_non_empty_name() -> None:
    identity = FaceIdentity(face_id=_face_id())
    with pytest.raises(ValidationError):
        identity.promote_to_known("  ")


def test_cannot_promote_known_identity() -> None:
    identity = FaceIdentity(face_id=_face_id())
    identity.promote_to_known("Alice")
    with pytest.raises(InvalidTransitionError):
        identity.promote_to_known("Bob")


def test_deactivate_sets_deleted_at() -> None:
    identity = FaceIdentity(face_id=_face_id())
    identity.deactivate()
    assert identity.is_active is False
    assert identity.deleted_at is not None


def test_cannot_promote_inactive_identity() -> None:
    identity = FaceIdentity(face_id=_face_id())
    identity.deactivate()
    with pytest.raises(InvalidTransitionError):
        identity.promote_to_known("Alice")


def test_known_identity_requires_display_name_at_creation() -> None:
    with pytest.raises(ValidationError):
        FaceIdentity(face_id=_face_id(), status="known")
