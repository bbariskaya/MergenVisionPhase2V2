"""Integration tests for Qdrant store."""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from mv_phase1_bulk.qdrant_store import QdrantStore


@pytest.mark.asyncio
async def test_upsert_and_retrieve(qdrant_store: QdrantStore) -> None:
    try:
        await qdrant_store.ensure_collection()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qdrant not available: {exc}")

    sample_id = str(uuid.uuid4())
    face_id = str(uuid.uuid4())
    embedding = np.ones(512, dtype=np.float32)

    await qdrant_store.upsert(sample_id, face_id, embedding)
    point = await qdrant_store.retrieve(sample_id)
    assert point is not None
    assert point.payload["face_id"] == face_id

    await qdrant_store.delete_best_effort(sample_id)
