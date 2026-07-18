"""MinIO store writing into Phase 2's existing object bucket.

Uploads follow the Phase 2 contract:
- Object key: ``faces/{face_id}/{sample_id}/aligned.jpg``
- Content type: ``image/jpeg``
- Metadata: ``x-amz-meta-sha256`` = hex digest of payload
- Existing objects are verified by size and SHA; conflicts fail-closed.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from minio import Minio
from minio.error import S3Error


@dataclass(frozen=True)
class ObjectStat:
    object_key: str
    size: int | None
    sha256: str | None


@dataclass(frozen=True)
class UploadResult:
    object_key: str
    etag: str | None
    sha256: str
    bytes_written: int


class MinioStore:
    """Bounded-concurrent MinIO uploader for aligned WebP face crops."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        secure: bool = False,
        max_concurrency: int = 32,
    ) -> None:
        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self._bucket_name = bucket_name
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._ensure_bucket_task: asyncio.Task[Any] | None = None

    async def _ensure_bucket(self) -> None:
        """Create bucket idempotently (once per store lifetime)."""
        if self._ensure_bucket_task is not None:
            await self._ensure_bucket_task
            return

        async def _create() -> None:
            try:
                if not self._client.bucket_exists(self._bucket_name):
                    self._client.make_bucket(self._bucket_name)
            except S3Error as exc:
                if exc.code not in ("BucketAlreadyExists", "BucketAlreadyOwnedByYou"):
                    raise

        self._ensure_bucket_task = asyncio.create_task(_create())
        await self._ensure_bucket_task

    def _object_key(self, face_id: str, sample_id: str) -> str:
        return f"faces/{face_id}/{sample_id}/aligned.jpg"

    def _stat_sync(self, object_key: str) -> ObjectStat | None:
        try:
            info = self._client.stat_object(self._bucket_name, object_key)
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                return None
            raise
        sha256 = info.metadata.get("sha256") if info.metadata else None
        if sha256 is None:
            sha256 = info.metadata.get("X-Amz-Meta-Sha256") if info.metadata else None
        return ObjectStat(
            object_key=object_key,
            size=info.size,
            sha256=sha256,
        )

    def _upload_sync(
        self,
        face_id: str,
        sample_id: str,
        data: bytes,
        content_type: str = "image/jpeg",
    ) -> UploadResult:
        sha256 = hashlib.sha256(data).hexdigest()
        object_key = self._object_key(face_id, sample_id)
        existing = self._stat_sync(object_key)
        if existing is not None:
            if existing.size != len(data):
                raise RuntimeError(f"BLOCKED_MINIO_CONFLICT: {object_key} exists with different size")
            if existing.sha256 is not None and existing.sha256 != sha256:
                raise RuntimeError(f"BLOCKED_MINIO_CONFLICT: {object_key} exists with different sha256")
            return UploadResult(
                object_key=object_key,
                etag=None,
                sha256=sha256,
                bytes_written=len(data),
            )
        result = self._client.put_object(
            self._bucket_name,
            object_key,
            BytesIO(data),
            length=len(data),
            content_type=content_type,
            metadata={"sha256": sha256},
        )
        return UploadResult(
            object_key=object_key,
            etag=result.etag,
            sha256=sha256,
            bytes_written=len(data),
        )

    async def upload_sample(
        self,
        face_id: str,
        sample_id: str,
        data: bytes,
        content_type: str = "image/jpeg",
    ) -> UploadResult:
        await self._ensure_bucket()
        async with self._semaphore:
            return await asyncio.to_thread(
                self._upload_sync,
                face_id,
                sample_id,
                data,
                content_type,
            )

    async def upload_many(
        self,
        items: list[tuple[str, str, bytes]],
        content_type: str = "image/jpeg",
    ) -> list[UploadResult | BaseException]:
        """Upload many samples concurrently with bounded parallelism.

        Returns a list aligned with ``items``; failed uploads are returned as
        exceptions so callers can decide whether to fail samples individually.
        """
        await self._ensure_bucket()

        async def _one(face_id: str, sample_id: str, data: bytes) -> UploadResult:
            async with self._semaphore:
                return await asyncio.to_thread(self._upload_sync, face_id, sample_id, data, content_type)

        return await asyncio.gather(
            *(_one(face_id, sample_id, data) for face_id, sample_id, data in items),
            return_exceptions=True,
        )

    async def stat(self, object_key: str) -> ObjectStat | None:
        await self._ensure_bucket()
        async with self._semaphore:
            return await asyncio.to_thread(self._stat_sync, object_key)

    async def delete_best_effort(self, object_key: str) -> None:
        """Best-effort cleanup; used during rollback."""
        try:
            await self._ensure_bucket()
            async with self._semaphore:
                await asyncio.to_thread(self._client.remove_object, self._bucket_name, object_key)
        except Exception:
            pass
