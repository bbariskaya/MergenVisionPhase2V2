"""Concurrent Unit of Work isolation integration tests."""

from __future__ import annotations

import asyncio

from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.domain.value_objects import BoundingBox
from tests.fixtures.embedding_fixtures import vector_a, vector_b

BBOX = BoundingBox(x=0, y=0, width=16, height=16)
MATCH_THRESHOLD = 0.95


async def test_same_service_instance_handles_two_concurrent_resolve_calls(
    lifecycle_service: IdentityStorageLifecycleService,
    crop_bytes: bytes,
) -> None:
    first, second = await asyncio.gather(
        lifecycle_service.resolve_or_create(
            crop_bytes=crop_bytes,
            embedding=vector_a(),
            bbox=BBOX,
            match_threshold=MATCH_THRESHOLD,
        ),
        lifecycle_service.resolve_or_create(
            crop_bytes=crop_bytes,
            embedding=vector_b(),
            bbox=BBOX,
            match_threshold=MATCH_THRESHOLD,
        ),
    )

    assert first.face_id != second.face_id
    assert first.status == "new_anonymous"
    assert second.status == "new_anonymous"

    from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
    from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork

    async with SqlAlchemyUnitOfWork(async_session_maker) as uow:
        processes = await uow.processes.list_by_status("processing")
        completed = await uow.processes.list_by_status("completed")

    assert len(processes) == 0
    assert len(completed) == 2
