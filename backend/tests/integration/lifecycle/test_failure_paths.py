"""Failure-path integration tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from app.application.ports.id_generator import IdGenerator
from app.application.ports.object_store import ObjectStore
from app.application.ports.vector_store import VectorCandidate, VectorStore
from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.domain.errors import IdentityResolutionError
from app.domain.value_objects import BoundingBox, FaceId, ObjectStat, SampleId
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from app.infrastructure.storage.minio_adapter import MinIOObjectStore
from app.infrastructure.uuid7 import Uuid7Generator
from app.infrastructure.vectors.qdrant_adapter import QdrantVectorStore
from tests.fixtures.embedding_fixtures import vector_a

BBOX = BoundingBox(x=0, y=0, width=16, height=16)
MATCH_THRESHOLD = 0.95


def _uow_factory() -> SqlAlchemyUnitOfWork:
    return SqlAlchemyUnitOfWork(async_session_maker)


@pytest.fixture
def id_generator() -> IdGenerator:
    return Uuid7Generator()


class FailingObjectStore(ObjectStore):
    async def upload(self, key: str, data: bytes, content_type: str) -> ObjectStat:
        raise RuntimeError("MinIO upload failure")

    async def upload_from_file(
        self, key: str, file_path: Path, content_type: str
    ) -> ObjectStat:
        raise RuntimeError("MinIO upload failure")

    async def copy(self, source_key: str, dest_key: str) -> None:
        raise RuntimeError("MinIO copy failure")

    async def stat(self, key: str) -> ObjectStat | None:
        return None

    async def delete(self, key: str) -> None:
        pass


class FailingVectorStore(VectorStore):
    async def upsert(
        self, sample_id: SampleId, face_id: FaceId, embedding: Sequence[float]
    ) -> None:
        raise RuntimeError("Qdrant upsert failure")

    async def query(self, embedding: Sequence[float], top_k: int) -> Sequence[VectorCandidate]:
        return []

    async def set_active(self, sample_id: SampleId, active: bool) -> None:
        pass

    async def delete(self, sample_id: SampleId) -> None:
        pass


async def test_minio_failure_no_completed_result(
    crop_bytes: bytes,
    id_generator: IdGenerator,
) -> None:
    service = IdentityStorageLifecycleService(
        unit_of_work_factory=_uow_factory,
        object_store=FailingObjectStore(),
        vector_store=QdrantVectorStore(),
        id_generator=id_generator,
    )

    with pytest.raises(IdentityResolutionError):
        await service.resolve_or_create(
            crop_bytes=crop_bytes,
            embedding=vector_a(),
            bbox=BBOX,
            match_threshold=MATCH_THRESHOLD,
        )

    async with SqlAlchemyUnitOfWork(async_session_maker) as uow:
        completed = await uow.processes.list_by_status("completed")
        assert len(completed) == 0

        failed = await uow.processes.list_by_status("failed")
        assert len(failed) == 1

        identities = await uow.face_identities.list_all()
        assert len(identities) == 1
        assert identities[0].is_active is False
        assert identities[0].deleted_at is not None


async def test_qdrant_failure_no_completed_result(
    crop_bytes: bytes,
    id_generator: IdGenerator,
) -> None:
    real_minio = MinIOObjectStore()
    service = IdentityStorageLifecycleService(
        unit_of_work_factory=_uow_factory,
        object_store=real_minio,
        vector_store=FailingVectorStore(),
        id_generator=id_generator,
    )

    with pytest.raises(IdentityResolutionError):
        await service.resolve_or_create(
            crop_bytes=crop_bytes,
            embedding=vector_a(),
            bbox=BBOX,
            match_threshold=MATCH_THRESHOLD,
        )

    async with SqlAlchemyUnitOfWork(async_session_maker) as uow:
        completed = await uow.processes.list_by_status("completed")
        assert len(completed) == 0

        failed = await uow.processes.list_by_status("failed")
        assert len(failed) == 1

        identities = await uow.face_identities.list_all()
        assert len(identities) == 1
        assert identities[0].is_active is False

    # Verify MinIO cleanup: bucket should contain no face-sample objects.
    objects = list(
        real_minio._client.list_objects(real_minio._bucket_name, prefix="faces/", recursive=True)
    )
    assert len(objects) == 0
