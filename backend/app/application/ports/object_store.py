"""Object store port for binary storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.domain.value_objects import ObjectStat


class ObjectStore(ABC):
    @abstractmethod
    async def upload(self, key: str, data: bytes, content_type: str) -> ObjectStat: ...

    @abstractmethod
    async def upload_from_file(
        self,
        key: str,
        file_path: Path,
        content_type: str,
    ) -> ObjectStat: ...

    @abstractmethod
    async def copy(self, source_key: str, dest_key: str) -> None: ...

    @abstractmethod
    async def stat(self, key: str) -> ObjectStat | None: ...

    @abstractmethod
    async def get(self, key: str) -> bytes | None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def presigned_get_url(
        self,
        key: str,
        expiry_seconds: int = 3600,
        response_content_type: str | None = None,
    ) -> str | None: ...
