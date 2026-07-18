"""Phase 2 Milestone 0.4 — delete/detail/history semantics on real PostgreSQL."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from uuid import UUID

import pytest

from app.application.ports.image_recognition import (
    ImageRecognitionEngine,
    NativeRecognitionResult,
)
from app.application.ports.unit_of_work import UnitOfWork
from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.application.services.image_recognition_service import ImageRecognitionService
from app.domain.entities.face_identity import FaceIdentity
from app.domain.entities.face_sample import FaceSample
from app.domain.entities.person import Person
from app.domain.entities.process_record import ProcessRecord
from app.domain.entities.recognition_result import RecognitionResult
from app.domain.value_objects import BoundingBox, FaceId, PersonId, ProcessId, ResultId, SampleId
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from app.infrastructure.storage.minio_adapter import MinIOObjectStore
from app.infrastructure.uuid7 import Uuid7Generator
from app.infrastructure.vectors.qdrant_adapter import QdrantVectorStore

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def uow() -> AsyncGenerator[UnitOfWork, None]:
    async with SqlAlchemyUnitOfWork(async_session_maker) as session:
        yield session


@pytest.fixture
def unit_of_work_factory():
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
def id_generator() -> Uuid7Generator:
    return Uuid7Generator()


class _FakeRecognitionEngine(ImageRecognitionEngine):
    async def detect_and_embed(self, image_bytes: bytes) -> NativeRecognitionResult:
        raise RuntimeError("fake engine: should not be called")


async def test_delete_faces_and_history_preserved(
    uow: UnitOfWork,
    unit_of_work_factory,
    object_store: MinIOObjectStore,
    vector_store: QdrantVectorStore,
    id_generator: Uuid7Generator,
) -> None:
    face_id = FaceId(UUID("018f1000-0000-7b0e-8000-000000000001"))
    sample_id = SampleId(UUID("018f1000-0000-7b0e-8000-000000000002"))
    process_id = ProcessId(UUID("018f1000-0000-7b0e-8000-000000000003"))
    recognition_id = UUID("018f1000-0000-7b0e-8000-000000000004")
    person_id = PersonId(UUID("018f1000-0000-7b0e-8000-000000000005"))

    # Seed active identity and sample.
    person = Person(
        person_id=person_id,
        display_name="Ada",
        person_metadata={},
    )
    identity = FaceIdentity(
        face_id=face_id,
        status="known",
        display_name="Ada",
        person_id=person_id,
    )
    sample = FaceSample(
        sample_id=sample_id,
        face_id=face_id,
        state="active",
        bucket=object_store._bucket_name,
        object_key=f"faces/{face_id}/{sample_id}/aligned.webp",
        is_active=True,
        created_at=datetime.now(UTC),
        activated_at=datetime.now(UTC),
    )
    process = ProcessRecord(
        process_id=process_id,
        process_type="image_recognize",
        status="completed",
        face_count=1,
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    result = RecognitionResult(
        result_id=ResultId(recognition_id),
        process_id=process_id,
        face_id=face_id,
        sample_id=sample_id,
        status="known",
        bounding_box=BoundingBox(x=10, y=20, width=30, height=40),
        match_confidence=0.88,
        created_at=datetime.now(UTC),
        metadata={},
    )

    await uow.people.add(person)
    await uow.flush()
    await uow.face_identities.add(identity)
    await uow.face_samples.add(sample)
    await uow.processes.add(process)
    await uow.recognition_results.add(result)
    await uow.commit()

    # Mirror sample vector in Qdrant so deactivation has something to flip.
    await vector_store.upsert(sample_id, face_id, [0.1] * 512)

    # Delete the identity.
    lifecycle = IdentityStorageLifecycleService(
        unit_of_work_factory=unit_of_work_factory,
        object_store=object_store,
        vector_store=vector_store,
        id_generator=id_generator,
    )
    await lifecycle.deactivate_identity(face_id)

    # Detail must return None (public 404 semantics).
    async with SqlAlchemyUnitOfWork(async_session_maker) as fresh_uow:
        detail = await fresh_uow.face_identities.get_active_by_id(face_id)
    assert detail is None

    # History must still contain the immutable recognition result.
    service = ImageRecognitionService(
        lifecycle_service=lifecycle,
        unit_of_work_factory=unit_of_work_factory,
        max_image_bytes=10_000_000,
        model_version="phase1-sprint-01-test",
        match_threshold=0.55,
        engine_factory=lambda: _FakeRecognitionEngine(),
    )
    history = await service.get_face_history(face_id)
    assert len(history) == 1
    assert history[0]["recognition_status"] == "known"

    # Sample must be inactive in PG.
    async with SqlAlchemyUnitOfWork(async_session_maker) as fresh_uow:
        updated_sample = await fresh_uow.face_samples.get_by_id(sample_id)
    assert updated_sample is not None
    assert updated_sample.is_active is False
    assert updated_sample.state == "inactive"

    # Qdrant point must be active=false.
    await vector_store.ensure_collection()
    point = await vector_store.client.retrieve(
        collection_name=vector_store._collection_name,
        ids=[str(sample_id)],
        with_payload=True,
    )
    assert len(point) == 1
    assert point[0].payload.get("active") is False
