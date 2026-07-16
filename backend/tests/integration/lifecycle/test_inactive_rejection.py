"""Inactive identity rejection integration tests."""

from __future__ import annotations

from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.domain.value_objects import BoundingBox
from tests.fixtures.embedding_fixtures import vector_a

BBOX = BoundingBox(x=0, y=0, width=16, height=16)
MATCH_THRESHOLD = 0.95


async def test_inactive_identity_not_returned_as_candidate(
    lifecycle_service: IdentityStorageLifecycleService,
    crop_bytes: bytes,
) -> None:
    first = await lifecycle_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    await lifecycle_service.deactivate_identity(first.face_id)

    second = await lifecycle_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    assert second.status == "new_anonymous"
    assert second.face_id != first.face_id
