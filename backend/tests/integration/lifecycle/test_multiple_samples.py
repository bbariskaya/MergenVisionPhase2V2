"""Multiple samples per identity integration tests."""

from __future__ import annotations

from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.domain.value_objects import BoundingBox
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from tests.fixtures.embedding_fixtures import vector_a

BBOX = BoundingBox(x=0, y=0, width=16, height=16)
MATCH_THRESHOLD = 0.95


async def test_multiple_samples_same_face(
    lifecycle_service: IdentityStorageLifecycleService,
    crop_bytes: bytes,
) -> None:
    first = await lifecycle_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    second_sample = await lifecycle_service.add_sample(
        face_id=first.face_id,
        crop_bytes=crop_bytes,
        embedding=vector_a(),
    )

    assert second_sample.face_id == first.face_id
    assert second_sample.sample_id != first.sample_id
    assert second_sample.state == "active"

    async with SqlAlchemyUnitOfWork(async_session_maker) as uow:
        active = await uow.face_samples.list_active_by_face_id(first.face_id)
        assert len(active) == 2
