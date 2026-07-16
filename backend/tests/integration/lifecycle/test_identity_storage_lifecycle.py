"""Core identity lifecycle integration tests."""

from __future__ import annotations

from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.domain.value_objects import BoundingBox
from tests.fixtures.embedding_fixtures import vector_a, vector_b

BBOX = BoundingBox(x=10, y=20, width=30, height=40)
MATCH_THRESHOLD = 0.95


async def test_vector_a_first_request_returns_new_anonymous(
    lifecycle_service: IdentityStorageLifecycleService,
    crop_bytes: bytes,
) -> None:
    outcome = await lifecycle_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    assert outcome.status == "new_anonymous"
    assert outcome.match_confidence == 0.0
    assert outcome.bounding_box == BBOX


async def test_vector_a_repeated_returns_anonymous_same_face(
    lifecycle_service: IdentityStorageLifecycleService,
    crop_bytes: bytes,
) -> None:
    first = await lifecycle_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    second = await lifecycle_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    assert second.face_id == first.face_id
    assert second.status == "anonymous"
    assert second.match_confidence > MATCH_THRESHOLD


async def test_orthogonal_vector_b_returns_new_anonymous_different_face(
    lifecycle_service: IdentityStorageLifecycleService,
    crop_bytes: bytes,
) -> None:
    first = await lifecycle_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    second = await lifecycle_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_b(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    assert second.status == "new_anonymous"
    assert second.face_id != first.face_id


async def test_enroll_preserves_face_id_and_later_known(
    lifecycle_service: IdentityStorageLifecycleService,
    crop_bytes: bytes,
) -> None:
    anonymous = await lifecycle_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    enrolled = await lifecycle_service.enroll_identity(
        face_id=anonymous.face_id,
        display_name="Alice",
        metadata={"department": "Engineering"},
    )

    assert enrolled.face_id == anonymous.face_id
    assert enrolled.status == "known"
    assert enrolled.display_name == "Alice"

    recognized = await lifecycle_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    assert recognized.face_id == anonymous.face_id
    assert recognized.status == "known"


async def test_old_result_snapshot_remains_new_anonymous(
    lifecycle_service: IdentityStorageLifecycleService,
    crop_bytes: bytes,
) -> None:
    from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
    from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork

    anonymous = await lifecycle_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )
    process_id = anonymous.process_id

    await lifecycle_service.enroll_identity(
        face_id=anonymous.face_id,
        display_name="Alice",
        metadata={},
    )

    async with SqlAlchemyUnitOfWork(async_session_maker) as uow:
        results = await uow.recognition_results.list_by_process_id(process_id)
        assert len(results) == 1
        assert results[0].status == "new_anonymous"
