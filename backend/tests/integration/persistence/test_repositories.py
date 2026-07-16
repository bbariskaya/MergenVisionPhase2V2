"""Integration tests for SQLAlchemy repository adapters."""

from __future__ import annotations

import pytest

from app.domain.entities.face_identity import FaceIdentity
from app.domain.entities.face_sample import FaceSample
from app.domain.entities.process_record import ProcessRecord
from app.domain.entities.recognition_result import RecognitionResult
from app.domain.value_objects import BoundingBox, FaceId, ProcessId, ResultId, SampleId
from app.infrastructure.config import settings
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from app.infrastructure.uuid7 import generate_uuid7

pytestmark = pytest.mark.asyncio(scope="session")


def _face_id() -> FaceId:
    return FaceId(generate_uuid7())


def _sample_id() -> SampleId:
    return SampleId(generate_uuid7())


def _process_id() -> ProcessId:
    return ProcessId(generate_uuid7())


def _result_id() -> ResultId:
    return ResultId(generate_uuid7())


async def test_face_identity_crud(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    face_id = _face_id()
    identity = FaceIdentity(face_id=face_id)

    async with unit_of_work:
        await unit_of_work.face_identities.add(identity)
        await unit_of_work.commit()

    async with unit_of_work:
        loaded = await unit_of_work.face_identities.get_by_id(face_id)
        assert loaded is not None
        assert loaded.status == "anonymous"

    async with unit_of_work:
        loaded = await unit_of_work.face_identities.get_by_id(face_id)
        assert loaded is not None
        loaded.promote_to_known("Alice", {"department": "Engineering"})
        await unit_of_work.face_identities.update(loaded)
        await unit_of_work.commit()

    async with unit_of_work:
        loaded = await unit_of_work.face_identities.get_by_id(face_id)
        assert loaded is not None
        assert loaded.status == "known"
        assert loaded.display_name == "Alice"


async def test_face_sample_crud(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    face_id = _face_id()
    sample_id = _sample_id()

    async with unit_of_work:
        await unit_of_work.face_identities.add(FaceIdentity(face_id=face_id))
        await unit_of_work.face_samples.add(FaceSample(sample_id=sample_id, face_id=face_id))
        await unit_of_work.commit()

    async with unit_of_work:
        loaded = await unit_of_work.face_samples.get_by_id(sample_id)
        assert loaded is not None
        assert loaded.state == "pending"
        assert loaded.is_active is False

    async with unit_of_work:
        loaded = await unit_of_work.face_samples.get_by_id(sample_id)
        assert loaded is not None
        loaded.mark_active(settings.minio_bucket_name, f"faces/{face_id}/{sample_id}/aligned.webp")
        await unit_of_work.face_samples.update(loaded)
        await unit_of_work.commit()

    async with unit_of_work:
        active = await unit_of_work.face_samples.list_active_by_face_id(face_id)
        assert len(active) == 1
        assert active[0].state == "active"


async def test_process_record_crud(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    process_id = _process_id()
    process = ProcessRecord(process_id=process_id, process_type="image_recognize")

    async with unit_of_work:
        await unit_of_work.processes.add(process)
        await unit_of_work.commit()

    async with unit_of_work:
        loaded = await unit_of_work.processes.get_by_id(process_id)
        assert loaded is not None
        assert loaded.status == "processing"

    async with unit_of_work:
        loaded = await unit_of_work.processes.get_by_id(process_id)
        assert loaded is not None
        loaded.complete(face_count=1, details={"face_ids": []})
        await unit_of_work.processes.update(loaded)
        await unit_of_work.commit()

    async with unit_of_work:
        loaded = await unit_of_work.processes.get_by_id(process_id)
        assert loaded is not None
        assert loaded.status == "completed"
        assert loaded.face_count == 1


async def test_recognition_result_crud(unit_of_work: SqlAlchemyUnitOfWork) -> None:
    face_id = _face_id()
    process_id = _process_id()
    result_id = _result_id()

    async with unit_of_work:
        await unit_of_work.face_identities.add(FaceIdentity(face_id=face_id))
        await unit_of_work.processes.add(
            ProcessRecord(process_id=process_id, process_type="image_recognize")
        )
        result = RecognitionResult(
            result_id=result_id,
            process_id=process_id,
            face_id=face_id,
            status="new_anonymous",
            bounding_box=BoundingBox(x=0, y=0, width=10, height=10),
            match_confidence=0.0,
        )
        await unit_of_work.recognition_results.add(result)
        await unit_of_work.commit()

    async with unit_of_work:
        results = await unit_of_work.recognition_results.list_by_process_id(process_id)
        assert len(results) == 1
        assert results[0].status == "new_anonymous"
