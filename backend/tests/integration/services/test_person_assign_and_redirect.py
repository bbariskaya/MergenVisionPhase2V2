"""Integration tests for Person directory, assign/redirect and canonical resolution."""

from __future__ import annotations

import pytest

from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.application.services.person_management_service import PersonManagementService
from app.domain.value_objects import BoundingBox, FaceId
from tests.fixtures.embedding_fixtures import vector_a, vector_b

pytestmark = pytest.mark.asyncio

BBOX = BoundingBox(x=10, y=20, width=30, height=40)
MATCH_THRESHOLD = 0.95


@pytest.fixture
def person_service(unit_of_work_factory) -> PersonManagementService:
    return PersonManagementService(unit_of_work_factory=unit_of_work_factory)


async def test_enroll_creates_new_person_and_face(
    lifecycle_service: IdentityStorageLifecycleService,
    person_service: PersonManagementService,
    crop_bytes: bytes,
) -> None:
    outcome = await lifecycle_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )
    face_id = outcome.face_id

    enrolled = await lifecycle_service.enroll_identity(
        face_id=face_id,
        display_name="Alice",
        metadata={"department": "Engineering"},
    )
    assert enrolled.status == "known"
    assert enrolled.person_id is not None
    assert enrolled.display_name == "Alice"

    person = await person_service.get_person(enrolled.person_id)
    assert person is not None
    assert person.display_name == "Alice"
    assert person.is_active is True


async def test_assign_second_face_to_existing_person_redirects(
    lifecycle_service: IdentityStorageLifecycleService,
    person_service: PersonManagementService,
    unit_of_work_factory: UnitOfWorkFactory,
    crop_bytes: bytes,
) -> None:
    first = await lifecycle_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )
    enrolled = await lifecycle_service.enroll_identity(
        face_id=first.face_id,
        display_name="Bob",
        metadata={},
    )
    canonical_face_id = enrolled.face_id
    person_id = enrolled.person_id

    second = await lifecycle_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_b(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )
    source_face_id = second.face_id

    assigned = await lifecycle_service.assign_identity_to_person(
        face_id=source_face_id,
        target_person_id=person_id,
    )
    assert assigned.is_active is False
    assert assigned.redirect_to_face_id == canonical_face_id

    person, faces = await person_service.get_person_with_faces(person_id)
    assert person is not None
    assert len(faces) == 1
    assert faces[0].face_id == canonical_face_id

    async with unit_of_work_factory() as uow:
        canonical = await uow.face_identities.get_canonical_by_id(FaceId(source_face_id))
    assert canonical is not None
    assert canonical.face_id == canonical_face_id
