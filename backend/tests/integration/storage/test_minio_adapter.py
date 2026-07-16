"""Integration tests for MinIO object-store adapter."""

from __future__ import annotations

import uuid
from pathlib import Path

from app.infrastructure.storage.minio_adapter import MinIOObjectStore

FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "valid_crop.webp"


async def test_upload_and_stat_valid_webp() -> None:
    store = MinIOObjectStore()
    data = FIXTURE_PATH.read_bytes()
    sample_id = uuid.uuid4()
    key = f"faces/{uuid.uuid4()}/{sample_id}/aligned.webp"

    stat = await store.upload(key, data, "image/webp")

    assert stat.key == key
    assert stat.size == len(data)

    fetched = await store.stat(key)
    assert fetched is not None
    assert fetched.size == len(data)

    await store.delete(key)
    assert await store.stat(key) is None
