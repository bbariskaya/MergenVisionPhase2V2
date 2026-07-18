"""Shared integration test configuration.

This module is loaded before any backend/tests/integration/*/conftest.py. It
validates the dedicated test environment and provides shared infrastructure
fixtures.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.support.resource_guard import assert_safe_test_environment

# Fail-fast if the process is not configured for the isolated test namespace.
assert_safe_test_environment()

from app.application.ports.id_generator import IdGenerator
from app.application.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory
from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from app.infrastructure.storage.minio_adapter import MinIOObjectStore
from app.infrastructure.vectors.qdrant_adapter import QdrantVectorStore


@pytest.fixture
def unit_of_work() -> UnitOfWork:
    return SqlAlchemyUnitOfWork(async_session_maker)


@pytest.fixture
def unit_of_work_factory() -> UnitOfWorkFactory:
    def _factory() -> UnitOfWork:
        return SqlAlchemyUnitOfWork(async_session_maker)

    return _factory


@pytest.fixture
def object_store() -> MinIOObjectStore:
    return MinIOObjectStore()


@pytest.fixture
def vector_store() -> QdrantVectorStore:
    return QdrantVectorStore()


@pytest.fixture
def id_generator() -> IdGenerator:
    from app.infrastructure.uuid7 import Uuid7Generator

    return Uuid7Generator()


@pytest.fixture
def lifecycle_service(
    unit_of_work_factory: UnitOfWorkFactory,
    object_store: MinIOObjectStore,
    vector_store: QdrantVectorStore,
    id_generator: IdGenerator,
) -> IdentityStorageLifecycleService:
    return IdentityStorageLifecycleService(
        unit_of_work_factory=unit_of_work_factory,
        object_store=object_store,
        vector_store=vector_store,
        id_generator=id_generator,
    )


async def _clean_stores_async(
    object_store: MinIOObjectStore,
    vector_store: QdrantVectorStore,
) -> None:
    """Remove all data from the dedicated test PG, MinIO and Qdrant resources."""
    import asyncpg

    # PostgreSQL
    url = os.environ["DATABASE_URL"].replace("postgresql+asyncpg", "postgresql")
    conn = await asyncpg.connect(url)
    try:
        # Delete in FK-safe order (children before parents).
        await conn.execute("DELETE FROM video_track_sample")
        await conn.execute("DELETE FROM video_timeline_chunk")
        await conn.execute("DELETE FROM appearance_interval")
        await conn.execute("DELETE FROM video_tracklet")
        await conn.execute("DELETE FROM video_track")
        await conn.execute("DELETE FROM process_event")
        await conn.execute("DELETE FROM video_job")
        await conn.execute("DELETE FROM video_asset")
        await conn.execute("DELETE FROM recognition_result")
        await conn.execute("DELETE FROM face_sample")
        await conn.execute("DELETE FROM face_identity")
        await conn.execute("DELETE FROM person")
        await conn.execute("DELETE FROM process_record")
        await conn.execute("DELETE FROM idempotency_record")
        await conn.execute("DELETE FROM outbox_event")
    finally:
        await conn.close()

    # MinIO
    bucket = os.environ["MINIO_BUCKET_NAME"]
    if object_store._client.bucket_exists(bucket):
        objects = object_store._client.list_objects(bucket, prefix="faces/", recursive=True)
        for obj in objects:
            object_store._client.remove_object(bucket, obj.object_name)

    # Qdrant
    await vector_store.client.delete_collection(
        collection_name=os.environ["QDRANT_COLLECTION_NAME"]
    )


@pytest.fixture(autouse=True)
async def _clean_integration_stores(
    object_store: MinIOObjectStore,
    vector_store: QdrantVectorStore,
) -> None:
    """Run store cleanup before each integration test."""
    await _clean_stores_async(object_store, vector_store)


@pytest.fixture
def crop_bytes() -> bytes:
    path = Path(__file__).parents[1] / "fixtures" / "valid_crop.webp"
    return path.read_bytes()
