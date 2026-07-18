"""Unit tests for Person aggregate root."""

import pytest

from app.domain.entities.person import Person
from app.domain.errors import InvalidTransitionError, ValidationError
from app.domain.value_objects import PersonId
from app.infrastructure.uuid7 import generate_uuid7


def _person_id() -> PersonId:
    return PersonId(generate_uuid7())


def test_person_requires_display_name() -> None:
    with pytest.raises(ValidationError):
        Person(person_id=_person_id(), display_name="  ")


def test_rename_updates_name_and_metadata() -> None:
    person = Person(person_id=_person_id(), display_name="Alice")
    person.rename("Alice Smith", {"department": "Engineering"})
    assert person.display_name == "Alice Smith"
    assert person.person_metadata == {"department": "Engineering"}
    assert person.version == 2


def test_cannot_rename_inactive_person() -> None:
    person = Person(person_id=_person_id(), display_name="Alice")
    person.deactivate()
    with pytest.raises(InvalidTransitionError):
        person.rename("Alice Smith")


def test_deactivate_sets_deleted_at() -> None:
    person = Person(person_id=_person_id(), display_name="Alice")
    person.deactivate()
    assert person.is_active is False
    assert person.deleted_at is not None
    assert person.version == 2
