"""MinIO object-store adapter."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any
from urllib.parse import urlparse

from minio import Minio
from minio.error import S3Error

from app.application.ports.object_store import ObjectStore
from app.domain.errors import ValidationError
from app.domain.value_objects import ObjectStat
from app.infrastructure.config import settings

_SHA256_META_KEY = "sha256"


class MinIOObjectStore(ObjectStore):
    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        secure: bool | None = None,
        bucket_name: str | None = None,
    ) -> None:
        endpoint = endpoint or settings.minio_endpoint
        access_key = access_key or settings.minio_access_key
        secret_key = secret_key or settings.minio_secret_key
        secure = secure if secure is not None else settings.minio_secure
        self._bucket_name = bucket_name or settings.minio_bucket_name

        parsed = urlparse(f"http://{endpoint}") if "://" not in endpoint else urlparse(endpoint)
        self._client = Minio(
            parsed.netloc or parsed.path,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    async def _ensure_bucket(self) -> None:
        def _check() -> bool:
            return self._client.bucket_exists(self._bucket_name)

        exists = await asyncio.to_thread(_check)
        if not exists:

            def _make() -> None:
                self._client.make_bucket(self._bucket_name)

            await asyncio.to_thread(_make)

    def _compute_sha256(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    async def upload(self, key: str, data: bytes, content_type: str) -> ObjectStat:
        await self._ensure_bucket()

        if not isinstance(data, bytes) or len(data) == 0:
            raise ValidationError("upload data must be non-empty bytes")

        sha256 = self._compute_sha256(data)

        # Idempotency: same key with same content/size is allowed.
        existing = await self.stat(key)
        if existing is not None:
            if existing.size != len(data):
                raise ValidationError(f"Object key {key} already exists with different size")
            if existing.sha256 is not None and existing.sha256 != sha256:
                raise ValidationError(f"Object key {key} already exists with different content")
            return existing

        def _put() -> None:
            from io import BytesIO

            self._client.put_object(
                self._bucket_name,
                key,
                BytesIO(data),
                length=len(data),
                content_type=content_type,
                metadata={_SHA256_META_KEY: sha256},
            )

        await asyncio.to_thread(_put)
        stat = await self.stat(key)
        if stat is None:
            raise RuntimeError(f"Upload succeeded but stat returned None for {key}")
        if stat.size != len(data):
            raise RuntimeError(f"Uploaded size mismatch for {key}")
        return stat

    async def stat(self, key: str) -> ObjectStat | None:
        await self._ensure_bucket()

        def _stat() -> Any:
            try:
                return self._client.stat_object(self._bucket_name, key)
            except S3Error as exc:
                if exc.code == "NoSuchKey":
                    return None
                raise

        result = await asyncio.to_thread(_stat)
        if result is None:
            return None
        metadata = result.metadata or {}
        # MinIO returns user metadata with the x-amz-meta- prefix.
        sha256 = metadata.get(f"x-amz-meta-{_SHA256_META_KEY}")
        if sha256 is None:
            sha256 = metadata.get(_SHA256_META_KEY)
        return ObjectStat(
            bucket=self._bucket_name,
            key=key,
            size=result.size,
            content_type=result.content_type,
            sha256=sha256,
        )

    async def delete(self, key: str) -> None:
        await self._ensure_bucket()

        def _remove() -> None:
            try:
                self._client.remove_object(self._bucket_name, key)
            except S3Error as exc:
                if exc.code != "NoSuchKey":
                    raise

        await asyncio.to_thread(_remove)
