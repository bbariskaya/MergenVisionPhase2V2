"""Unit of Work port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from app.application.ports.repositories import (
    FaceIdentityRepository,
    FaceSampleRepository,
    ProcessRepository,
    RecognitionResultRepository,
)


class UnitOfWork(ABC):
    face_identities: FaceIdentityRepository
    face_samples: FaceSampleRepository
    processes: ProcessRepository
    recognition_results: RecognitionResultRepository

    @abstractmethod
    async def __aenter__(self) -> Self: ...

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...

    @abstractmethod
    async def commit(self) -> None: ...

    @abstractmethod
    async def rollback(self) -> None: ...
