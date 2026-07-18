"""Integration tests for PostgreSQL store."""

from __future__ import annotations

import uuid

import pytest
from mv_phase1_bulk.postgres_store import PostgresStore
from mv_phase1_bulk.types import FaceRecord, PersonRecord, SampleRecord


async def _ping(store: PostgresStore) -> None:
    async with store.engine.connect() as conn:
        await conn.execute("SELECT 1")


@pytest.mark.asyncio
async def test_upsert_and_activate_sample(pg_store: PostgresStore) -> None:
    try:
        await _ping(pg_store)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PostgreSQL not available: {exc}")

    person_id = str(uuid.uuid4())
    face_id = str(uuid.uuid4())
    sample_id = str(uuid.uuid4())

    person = PersonRecord(person_id=person_id, display_name="int-test")
    face = FaceRecord(face_id=face_id, person_id=person_id)
    sample = SampleRecord(sample_id=sample_id, face_id=face_id)

    await pg_store.prepare_enrollment([person], [face], [sample])

    loaded = await pg_store.get_sample(sample_id)
    assert loaded is not None
    assert loaded.state == "pending"

    await pg_store.activate_samples_tx([(sample_id, "test-bucket", f"faces/{face_id}/{sample_id}/original.jpg")])
    activated = await pg_store.get_sample(sample_id)
    assert activated is not None
    assert activated.state == "active"
    assert activated.is_active is True
