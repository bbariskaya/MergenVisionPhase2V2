"""Vector store port for embedding similarity search."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.domain.value_objects import FaceId, SampleId


class VectorCandidate:
    def __init__(self, sample_id: SampleId, face_id: FaceId, score: float) -> None:
        self.sample_id = sample_id
        self.face_id = face_id
        self.score = score


class VectorStore(ABC):
    @abstractmethod
    async def upsert(
        self,
        sample_id: SampleId,
        face_id: FaceId,
        embedding: Sequence[float],
    ) -> None: ...

    @abstractmethod
    async def query(
        self,
        embedding: Sequence[float],
        top_k: int,
    ) -> Sequence[VectorCandidate]: ...

    @abstractmethod
    async def set_active(
        self,
        sample_id: SampleId,
        active: bool,
    ) -> None: ...

    @abstractmethod
    async def delete(self, sample_id: SampleId) -> None: ...
