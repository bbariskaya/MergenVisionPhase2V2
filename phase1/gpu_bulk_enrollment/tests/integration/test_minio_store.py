"""Integration tests for MinIO store."""

from __future__ import annotations

import uuid

import pytest
from mv_phase1_bulk.minio_store import MinioStore


@pytest.mark.asyncio
async def test_upload_and_stat(minio_store: MinioStore) -> None:
    try:
        await minio_store._ensure_bucket()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"MinIO not available: {exc}")

    face_id = str(uuid.uuid4())
    sample_id = str(uuid.uuid4())
    data = b"fake-jpeg-bytes"

    result = await minio_store.upload_sample(face_id, sample_id, data, content_type="image/jpeg")
    assert result.object_key == f"faces/{face_id}/{sample_id}/original.jpg"

    stat = await minio_store.stat(result.object_key)
    assert stat is not None
    assert stat.size == len(data)

    await minio_store.delete_best_effort(result.object_key)
