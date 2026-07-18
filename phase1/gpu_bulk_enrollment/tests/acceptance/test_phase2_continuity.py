"""Black-box continuity test: one persisted sample must exist in all three stores."""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from mv_phase1_bulk.identities import EnrolledSample, SubjectBundle
from mv_phase1_bulk.minio_store import MinioStore
from mv_phase1_bulk.persistence import PersistenceOrchestrator
from mv_phase1_bulk.postgres_store import PostgresStore
from mv_phase1_bulk.qdrant_store import QdrantStore
from mv_phase1_bulk.types import FaceRecord, PersonRecord, SampleRecord
from sqlalchemy import text


async def _check_services(pg_store: PostgresStore, minio_store: MinioStore, qdrant_store: QdrantStore) -> None:
    try:
        async with pg_store.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PostgreSQL not available: {exc}")

    try:
        await minio_store._ensure_bucket()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"MinIO not available: {exc}")

    try:
        await qdrant_store.ensure_collection()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qdrant not available: {exc}")


@pytest.mark.asyncio
async def test_end_to_end_persisted_sample_is_continuous(
    pg_store: PostgresStore,
    minio_store: MinioStore,
    qdrant_store: QdrantStore,
) -> None:
    await _check_services(pg_store, minio_store, qdrant_store)

    person_id = str(uuid.uuid4())
    face_id = str(uuid.uuid4())
    sample_id = str(uuid.uuid4())

    person = PersonRecord(person_id=person_id, display_name="continuity-test")
    face = FaceRecord(face_id=face_id, person_id=person_id)
    sample = SampleRecord(sample_id=sample_id, face_id=face_id)
    bundle = SubjectBundle(
        person=person,
        face=face,
        samples=[
            EnrolledSample(
                sample_record=sample,
                image_sha256="sha",
                image_bytes=b"jpeg-bytes",
                embedding=np.ones(512, dtype=np.float32),
            )
        ],
    )

    orchestrator = PersistenceOrchestrator(
        postgres=pg_store,
        minio=minio_store,
        qdrant=qdrant_store,
        model_version="mv1",
    )
    result = await orchestrator.persist_bundle(bundle)

    assert len(result.persisted) == 1
    persisted = result.persisted[0]

    # PostgreSQL: sample active with object key.
    db_sample = await pg_store.get_sample(persisted.sample_id)
    assert db_sample is not None
    assert db_sample.state == "active"
    assert db_sample.is_active is True
    assert db_sample.object_key == persisted.object_key

    # MinIO: object exists.
    stat = await minio_store.stat(persisted.object_key)
    assert stat is not None
    assert stat.size == len(b"jpeg-bytes")

    # Qdrant: point exists and active.
    point = await qdrant_store.retrieve(persisted.sample_id)
    assert point is not None
    assert point.payload["face_id"] == face_id
    assert point.payload["active"] is True
