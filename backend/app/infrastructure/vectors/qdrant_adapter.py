"""Qdrant vector-store adapter."""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.application.ports.vector_store import VectorCandidate, VectorStore
from app.domain.errors import ValidationError
from app.domain.value_objects import FaceId, SampleId
from app.infrastructure.config import settings

DIMENSION = 512


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        url: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._url = url or settings.qdrant_url
        self._collection_name = collection_name or settings.qdrant_collection_name
        self._client: AsyncQdrantClient | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(url=self._url)
        return self._client

    def _validate_embedding(self, embedding: Sequence[float]) -> None:
        if len(embedding) != DIMENSION:
            raise ValidationError(f"Embedding must have length {DIMENSION}")
        if not all(math.isfinite(v) for v in embedding):
            raise ValidationError("Embedding values must be finite")
        norm = math.sqrt(sum(v * v for v in embedding))
        if norm == 0.0:
            raise ValidationError("Embedding norm must be non-zero")

    async def _ensure_collection(self) -> None:
        collections = await self.client.get_collections()
        if any(c.name == self._collection_name for c in collections.collections):
            return
        await self.client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE),
        )

    async def upsert(
        self,
        sample_id: SampleId,
        face_id: FaceId,
        embedding: Sequence[float],
    ) -> None:
        self._validate_embedding(embedding)
        await self._ensure_collection()
        await self.client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(
                    id=str(sample_id),
                    vector=list(embedding),
                    payload={"face_id": str(face_id), "active": True},
                )
            ],
            wait=True,
        )

    async def query(
        self,
        embedding: Sequence[float],
        top_k: int,
    ) -> list[VectorCandidate]:
        self._validate_embedding(embedding)
        await self._ensure_collection()
        result = await self.client.query_points(
            collection_name=self._collection_name,
            query=list(embedding),
            query_filter=Filter(must=[FieldCondition(key="active", match=MatchValue(value=True))]),
            limit=top_k,
            with_payload=True,
        )
        candidates: list[VectorCandidate] = []
        for point in result.points:
            payload = point.payload or {}
            face_id_str = payload.get("face_id")
            if face_id_str is None:
                continue
            candidates.append(
                VectorCandidate(
                    sample_id=SampleId(uuid.UUID(str(point.id))),
                    face_id=FaceId(uuid.UUID(face_id_str)),
                    score=point.score,
                )
            )
        return candidates

    async def set_active(
        self,
        sample_id: SampleId,
        active: bool,
    ) -> None:
        await self._ensure_collection()
        await self.client.set_payload(
            collection_name=self._collection_name,
            points=[str(sample_id)],
            payload={"active": active},
            wait=True,
        )

    async def delete(self, sample_id: SampleId) -> None:
        await self._ensure_collection()
        await self.client.delete(
            collection_name=self._collection_name,
            points_selector=[str(sample_id)],
            wait=True,
        )
