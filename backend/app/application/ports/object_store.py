"""Object store port for face crop binary storage."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.value_objects import ObjectStat


class ObjectStore(ABC):
    @abstractmethod
    async def upload(self, key: str, data: bytes, content_type: str) -> ObjectStat: ...

    @abstractmethod
    async def stat(self, key: str) -> ObjectStat | None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...
