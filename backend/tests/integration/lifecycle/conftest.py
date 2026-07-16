"""Lifecycle integration test configuration and fixtures."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from urllib.parse import urlparse

import pytest

from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.storage.minio_adapter import MinIOObjectStore
from app.infrastructure.vectors.qdrant_adapter import QdrantVectorStore

pytestmark = pytest.mark.asyncio(scope="session")


async def _clean_stores_async() -> None:
    """Remove all data from PG, MinIO and Qdrant."""
    import asyncpg
    from minio import Minio
    from qdrant_client import AsyncQdrantClient

    from app.infrastructure.config import settings

    # PostgreSQL
    url = os.environ.get("DATABASE_URL", settings.database_url).replace(
        "postgresql+asyncpg", "postgresql"
    )
    conn = await asyncpg.connect(url)
    try:
        await conn.execute("DELETE FROM recognition_result")
        await conn.execute("DELETE FROM face_sample")
        await conn.execute("DELETE FROM process_record")
        await conn.execute("DELETE FROM face_identity")
    finally:
        await conn.close()

    # MinIO
    parsed = urlparse(f"http://{settings.minio_endpoint}")
    minio_client = Minio(
        parsed.netloc or parsed.path,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    if minio_client.bucket_exists(settings.minio_bucket_name):
        objects = minio_client.list_objects(
            settings.minio_bucket_name, prefix="faces/", recursive=True
        )
        for obj in objects:
            minio_client.remove_object(settings.minio_bucket_name, obj.object_name)

    # Qdrant
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        collections = await qdrant.get_collections()
        if any(c.name == settings.qdrant_collection_name for c in collections.collections):
            await qdrant.delete_collection(collection_name=settings.qdrant_collection_name)
    finally:
        await qdrant.close()


@pytest.fixture(autouse=True)
def _clean_lifecycle_stores() -> None:
    """Run store cleanup in an isolated event loop before each lifecycle test."""
    asyncio.run(_clean_stores_async())


@pytest.fixture
def crop_bytes() -> bytes:
    path = Path(__file__).parents[2] / "fixtures" / "valid_crop.webp"
    return path.read_bytes()


@pytest.fixture
def lifecycle_service() -> IdentityStorageLifecycleService:
    from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork

    return IdentityStorageLifecycleService(
        unit_of_work=SqlAlchemyUnitOfWork(async_session_maker),
        object_store=MinIOObjectStore(),
        vector_store=QdrantVectorStore(),
    )
