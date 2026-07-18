"""Unit tests for FaceIdentity state transitions including redirects."""

import pytest

from app.domain.entities.face_identity import FaceIdentity
from app.domain.errors import InvalidTransitionError, ValidationError
from app.domain.value_objects import FaceId, PersonId
from app.infrastructure.uuid7 import generate_uuid7


def _face_id() -> FaceId:
    return FaceId(generate_uuid7())


def _person_id() -> PersonId:
    return PersonId(generate_uuid7())


def test_new_identity_defaults_to_anonymous() -> None:
    identity = FaceIdentity(face_id=_face_id())
    assert identity.status == "anonymous"
    assert identity.is_active is True
    assert identity.display_name is None
    assert identity.person_id is None
    assert identity.redirect_to_face_id is None


def test_promote_anonymous_to_known() -> None:
    identity = FaceIdentity(face_id=_face_id())
    person_id = _person_id()
    identity.promote_to_known(person_id, "Alice", {"department": "Engineering"})
    assert identity.status == "known"
    assert identity.person_id == person_id
    assert identity.display_name == "Alice"
    assert identity.identity_metadata == {"department": "Engineering"}
    assert identity.version == 2


def test_promote_requires_non_empty_name() -> None:
    identity = FaceIdentity(face_id=_face_id())
    with pytest.raises(ValidationError):
        identity.promote_to_known(_person_id(), "  ")


def test_promote_requires_person_id() -> None:
    identity = FaceIdentity(face_id=_face_id())
    with pytest.raises(ValidationError):
        identity.promote_to_known(None, "Alice")  # type: ignore[arg-type]


def test_cannot_promote_known_identity() -> None:
    identity = FaceIdentity(face_id=_face_id())
    identity.promote_to_known(_person_id(), "Alice")
    with pytest.raises(InvalidTransitionError):
        identity.promote_to_known(_person_id(), "Bob")


def test_deactivate_sets_deleted_at() -> None:
    identity = FaceIdentity(face_id=_face_id())
    identity.deactivate()
    assert identity.is_active is False
    assert identity.deleted_at is not None


def test_cannot_promote_inactive_identity() -> None:
    identity = FaceIdentity(face_id=_face_id())
    identity.deactivate()
    with pytest.raises(InvalidTransitionError):
        identity.promote_to_known(_person_id(), "Alice")


def test_known_identity_requires_person_id_and_display_name_at_creation() -> None:
    with pytest.raises(ValidationError):
        FaceIdentity(face_id=_face_id(), status="known")
    with pytest.raises(ValidationError):
        FaceIdentity(face_id=_face_id(), status="known", display_name="Alice")


def test_assign_to_person_redirects_source() -> None:
    source = FaceIdentity(face_id=_face_id())
    target = FaceIdentity(face_id=_face_id())
    target.promote_to_known(_person_id(), "Bob")
    source.assign_to_person(target.person_id, target.face_id)

    assert source.is_active is False
    assert source.redirect_to_face_id == target.face_id
    assert source.person_id is None
    assert source.display_name is None


def test_assign_to_self_is_rejected() -> None:
    identity = FaceIdentity(face_id=_face_id())
    with pytest.raises(ValidationError):
        identity.assign_to_person(_person_id(), identity.face_id)


def test_assign_redirected_identity_is_rejected() -> None:
    source = FaceIdentity(face_id=_face_id())
    target = FaceIdentity(face_id=_face_id())
    target.promote_to_known(_person_id(), "Bob")
    source.assign_to_person(target.person_id, target.face_id)
    with pytest.raises(InvalidTransitionError):
        source.assign_to_person(target.person_id, target.face_id)


def test_canonical_face_id_follows_redirect() -> None:
    source = FaceIdentity(face_id=_face_id())
    target = FaceIdentity(face_id=_face_id())
    target.promote_to_known(_person_id(), "Bob")
    assert source.canonical_face_id == source.face_id
    source.assign_to_person(target.person_id, target.face_id)
    assert source.canonical_face_id == target.face_id
