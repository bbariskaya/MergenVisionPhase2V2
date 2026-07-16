"""Integration tests for MinIO object-store adapter."""

from __future__ import annotations

from pathlib import Path

from app.infrastructure.storage.minio_adapter import MinIOObjectStore
from app.infrastructure.uuid7 import generate_uuid7

FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "valid_crop.webp"


async def test_upload_and_stat_valid_webp() -> None:
    store = MinIOObjectStore()
    data = FIXTURE_PATH.read_bytes()
    face_id = generate_uuid7()
    sample_id = generate_uuid7()
    key = f"faces/{face_id}/{sample_id}/aligned.webp"

    stat = await store.upload(key, data, "image/webp")

    assert stat.bucket == store._bucket_name
    assert stat.key == key
    assert stat.size == len(data)
    assert stat.sha256 is not None

    fetched = await store.stat(key)
    assert fetched is not None
    assert fetched.size == len(data)
    assert fetched.sha256 == stat.sha256

    await store.delete(key)
    assert await store.stat(key) is None
