"""MinIO object-store adapter."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from minio import Minio
from minio.commonconfig import CopySource
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

    async def check_access(self) -> None:
        """Verify that the configured bucket is reachable.

        Creates the bucket if it does not exist, proving write access.
        """
        await self._ensure_bucket()

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

    async def upload_from_file(
        self,
        key: str,
        file_path: Path,
        content_type: str,
    ) -> ObjectStat:
        await self._ensure_bucket()

        path = Path(file_path)
        size = path.stat().st_size
        if size <= 0:
            raise ValidationError("upload file must be non-empty")

        sha256 = await asyncio.to_thread(self._compute_file_sha256, path)

        existing = await self.stat(key)
        if existing is not None:
            if existing.size != size:
                raise ValidationError(f"Object key {key} already exists with different size")
            if existing.sha256 is not None and existing.sha256 != sha256:
                raise ValidationError(f"Object key {key} already exists with different content")
            return existing

        def _put() -> None:
            self._client.fput_object(
                self._bucket_name,
                key,
                str(path),
                content_type=content_type,
                metadata={_SHA256_META_KEY: sha256},
            )

        await asyncio.to_thread(_put)
        stat = await self.stat(key)
        if stat is None:
            raise RuntimeError(f"Upload succeeded but stat returned None for {key}")
        if stat.size != size:
            raise RuntimeError(f"Uploaded size mismatch for {key}")
        return stat

    def _compute_file_sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    async def copy(self, source_key: str, dest_key: str) -> None:
        await self._ensure_bucket()

        def _copy() -> None:
            self._client.copy_object(
                self._bucket_name,
                dest_key,
                CopySource(self._bucket_name, source_key),
            )

        await asyncio.to_thread(_copy)

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

    async def get(self, key: str) -> bytes | None:
        """Return the object's bytes, or None if it does not exist."""
        await self._ensure_bucket()

        def _read() -> bytes:
            response = self._client.get_object(self._bucket_name, key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()

        try:
            return await asyncio.to_thread(_read)
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                return None
            raise

    async def presigned_get_url(
        self,
        key: str,
        expiry_seconds: int = 3600,
        response_content_type: str | None = None,
    ) -> str | None:
        """Generate a temporary signed URL for a private object."""
        await self._ensure_bucket()

        def _url() -> str:
            response_headers: dict[str, str | list[str] | tuple[str]] | None = (
                {"response-content-type": response_content_type}
                if response_content_type
                else None
            )
            return self._client.presigned_get_object(
                self._bucket_name,
                key,
                expires=timedelta(seconds=expiry_seconds),
                response_headers=response_headers,
            )

        try:
            return await asyncio.to_thread(_url)
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                return None
            raise
