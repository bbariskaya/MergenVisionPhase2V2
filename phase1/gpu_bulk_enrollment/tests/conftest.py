"""Shared test fixtures for all Phase 1 bulk enrollment tests."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
from mv_phase1_bulk.config import get_settings
from mv_phase1_bulk.minio_store import MinioStore
from mv_phase1_bulk.postgres_store import PostgresStore
from mv_phase1_bulk.qdrant_store import QdrantStore


def _require_env(monkeypatch: Any) -> None:
    """Populate required environment variables so Settings loads in tests."""
    defaults: dict[str, str] = {
        "DATABASE_URL": os.environ.get("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test"),
        "MV_MINIO_ENDPOINT": os.environ.get("MV_MINIO_ENDPOINT", "localhost:9000"),
        "MV_MINIO_ACCESS_KEY": os.environ.get("MV_MINIO_ACCESS_KEY", "test"),
        "MV_MINIO_SECRET_KEY": os.environ.get("MV_MINIO_SECRET_KEY", "test"),
        "MV_MINIO_BUCKET_NAME": os.environ.get("MV_MINIO_BUCKET_NAME", "test"),
        "MV_QDRANT_URL": os.environ.get("MV_QDRANT_URL", "http://localhost:6333"),
        "MV_PHASE1_BULK_ID_HMAC_KEY": os.environ.get("MV_PHASE1_BULK_ID_HMAC_KEY", "test-key"),
    }
    for key, value in defaults.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def event_loop() -> Any:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def settings(monkeypatch: Any) -> Any:
    _require_env(monkeypatch)
    return get_settings()


@pytest.fixture
async def pg_store(settings: Any) -> PostgresStore:
    store = PostgresStore(settings.database_url)
    try:
        await store.connect()
    except ModuleNotFoundError as exc:
        pytest.skip(f"PostgreSQL driver missing: {exc}")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PostgreSQL not available: {exc}")
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture
async def minio_store(settings: Any) -> MinioStore:
    store = MinioStore(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket_name=settings.minio_bucket_name,
        secure=settings.minio_secure,
    )
    yield store


@pytest.fixture
async def qdrant_store(settings: Any) -> QdrantStore:
    store = QdrantStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
        model_version=settings.model_version,
    )
    yield store
