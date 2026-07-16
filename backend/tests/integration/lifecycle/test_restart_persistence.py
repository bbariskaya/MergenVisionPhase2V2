"""Restart persistence integration test."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.domain.value_objects import BoundingBox
from app.infrastructure.config import settings
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from app.infrastructure.storage.minio_adapter import MinIOObjectStore
from app.infrastructure.uuid7 import Uuid7Generator
from app.infrastructure.vectors.qdrant_adapter import QdrantVectorStore
from tests.fixtures.embedding_fixtures import vector_a

BBOX = BoundingBox(x=0, y=0, width=16, height=16)
MATCH_THRESHOLD = 0.95

REPO_ROOT = Path(__file__).parents[4]
COMPOSE_FILE = REPO_ROOT / "docker-compose.test.yml"


def _uow_factory() -> SqlAlchemyUnitOfWork:
    return SqlAlchemyUnitOfWork(async_session_maker)


@pytest.fixture
def id_generator() -> Uuid7Generator:
    return Uuid7Generator()


def _compose(args: list[str]) -> None:
    cmd = [
        "docker",
        "compose",
        "-p",
        "mergenvision-s01-test",
        "-f",
        str(COMPOSE_FILE),
        *args,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


async def test_data_survives_restart(
    crop_bytes: bytes,
    id_generator: Uuid7Generator,
) -> None:
    service = IdentityStorageLifecycleService(
        unit_of_work_factory=_uow_factory,
        object_store=MinIOObjectStore(),
        vector_store=QdrantVectorStore(),
        id_generator=id_generator,
    )

    first = await service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )
    assert first.sample_id is not None

    _compose(["restart", "postgres-test", "minio-test", "qdrant-test"])
    _compose(["up", "-d", "--wait"])

    # Re-create service to pick up fresh connections after restart.
    service_after = IdentityStorageLifecycleService(
        unit_of_work_factory=_uow_factory,
        object_store=MinIOObjectStore(),
        vector_store=QdrantVectorStore(),
        id_generator=id_generator,
    )

    # Verify PostgreSQL identity/sample/process/result persisted.
    async with SqlAlchemyUnitOfWork(async_session_maker) as uow:
        identity = await uow.face_identities.get_by_id(first.face_id)
        assert identity is not None
        assert identity.is_active is True

        sample = await uow.face_samples.get_by_id(first.sample_id)
        assert sample is not None
        assert sample.state == "active"
        assert sample.bucket == settings.minio_bucket_name
        assert sample.object_key is not None

        process = await uow.processes.get_by_id(first.process_id)
        assert process is not None
        assert process.status == "completed"

        results = await uow.recognition_results.list_by_process_id(first.process_id)
        assert len(results) == 1
        assert results[0].status == "new_anonymous"

    # Verify MinIO object persisted.
    minio_store = MinIOObjectStore()
    stat = await minio_store.stat(sample.object_key)
    assert stat is not None
    assert stat.size == len(crop_bytes)

    # Verify Qdrant point persisted and remains searchable.
    qdrant_store = QdrantVectorStore()
    query_results = await qdrant_store.query(vector_a(), top_k=1)
    assert len(query_results) == 1
    assert query_results[0].sample_id == first.sample_id
    assert query_results[0].face_id == first.face_id

    # Recognition after restart should match the same face.
    second = await service_after.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    assert second.face_id == first.face_id
    assert second.status == "anonymous"
