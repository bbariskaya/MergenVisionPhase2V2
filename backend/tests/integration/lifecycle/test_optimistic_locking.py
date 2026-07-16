"""Optimistic-locking integration tests against real PostgreSQL."""

from __future__ import annotations

import asyncio

import pytest

from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.domain.errors import ConcurrentUpdateError
from app.domain.value_objects import BoundingBox
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from tests.fixtures.embedding_fixtures import vector_a

BBOX = BoundingBox(x=0, y=0, width=16, height=16)
MATCH_THRESHOLD = 0.95


async def test_concurrent_enrollments_one_wins_with_optimistic_lock(
    lifecycle_service: IdentityStorageLifecycleService,
    crop_bytes: bytes,
) -> None:
    anonymous = await lifecycle_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    results = await asyncio.gather(
        lifecycle_service.enroll_identity(
            face_id=anonymous.face_id,
            display_name="First",
            metadata={},
        ),
        lifecycle_service.enroll_identity(
            face_id=anonymous.face_id,
            display_name="Second",
            metadata={},
        ),
        return_exceptions=True,
    )

    winners = [r for r in results if not isinstance(r, BaseException)]
    losers = [r for r in results if isinstance(r, BaseException)]

    assert len(winners) == 1
    assert winners[0].status == "known"

    assert len(losers) == 1
    assert isinstance(losers[0], ConcurrentUpdateError)

    async with SqlAlchemyUnitOfWork(async_session_maker) as uow:
        identity = await uow.face_identities.get_by_id(anonymous.face_id)

    assert identity is not None
    assert identity.status == "known"
    # Version must have been incremented exactly once by the winning writer.
    assert identity.version == 2


async def test_update_with_expected_version_rejects_stale_version(
    lifecycle_service: IdentityStorageLifecycleService,
    crop_bytes: bytes,
) -> None:
    anonymous = await lifecycle_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    async with SqlAlchemyUnitOfWork(async_session_maker) as uow:
        identity = await uow.face_identities.get_by_id(anonymous.face_id)
        assert identity is not None
        stale_version = identity.version

    # Enroll via service to bump the version in the database.
    await lifecycle_service.enroll_identity(
        face_id=anonymous.face_id,
        display_name="Alice",
        metadata={},
    )

    # Attempt to apply a mutation using the now-stale expected version.
    async with SqlAlchemyUnitOfWork(async_session_maker) as uow:
        identity = await uow.face_identities.get_by_id(anonymous.face_id)
        assert identity is not None
        identity.deactivate()

        with pytest.raises(ConcurrentUpdateError):
            await uow.face_identities.update_with_expected_version(
                identity,
                stale_version,
            )
