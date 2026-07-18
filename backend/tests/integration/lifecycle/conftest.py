"""Lifecycle integration test configuration and fixtures."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from tests.support.resource_guard import assert_safe_test_environment

assert_safe_test_environment()

from app.application.ports.id_generator import IdGenerator
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.infrastructure.storage.minio_adapter import MinIOObjectStore
from app.infrastructure.vectors.qdrant_adapter import QdrantVectorStore

pytestmark = pytest.mark.asyncio(scope="session")


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
        await conn.execute("DELETE FROM recognition_result")
        await conn.execute("DELETE FROM face_sample")
        await conn.execute("DELETE FROM process_record")
        await conn.execute("DELETE FROM face_identity")
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
def _clean_lifecycle_stores(
    object_store: MinIOObjectStore,
    vector_store: QdrantVectorStore,
) -> None:
    """Run store cleanup in an isolated event loop before each lifecycle test."""
    asyncio.run(_clean_stores_async(object_store, vector_store))


@pytest.fixture
def crop_bytes() -> bytes:
    path = Path(__file__).parents[2] / "fixtures" / "valid_crop.webp"
    return path.read_bytes()


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
