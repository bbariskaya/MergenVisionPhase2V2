"""Restart persistence integration test."""

from __future__ import annotations

import subprocess

from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.domain.value_objects import BoundingBox
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from app.infrastructure.storage.minio_adapter import MinIOObjectStore
from app.infrastructure.vectors.qdrant_adapter import QdrantVectorStore
from tests.fixtures.embedding_fixtures import vector_a

BBOX = BoundingBox(x=0, y=0, width=16, height=16)
MATCH_THRESHOLD = 0.95


async def test_data_survives_restart(
    crop_bytes: bytes,
) -> None:
    service = IdentityStorageLifecycleService(
        unit_of_work=SqlAlchemyUnitOfWork(async_session_maker),
        object_store=MinIOObjectStore(),
        vector_store=QdrantVectorStore(),
    )

    first = await service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    subprocess.run(
        ["docker", "compose", "restart", "postgres", "minio", "qdrant"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["docker", "compose", "up", "-d", "--wait"],
        check=True,
        capture_output=True,
    )

    # Re-create service to pick up fresh connections after restart.
    service_after = IdentityStorageLifecycleService(
        unit_of_work=SqlAlchemyUnitOfWork(async_session_maker),
        object_store=MinIOObjectStore(),
        vector_store=QdrantVectorStore(),
    )

    second = await service_after.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    assert second.face_id == first.face_id
    assert second.status == "anonymous"
