"""Qdrant vector-store adapter."""

from __future__ import annotations

import asyncio
import math
import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.application.ports.vector_store import VectorCandidate, VectorStore
from app.domain.errors import ValidationError
from app.domain.value_objects import FaceId, SampleId
from app.infrastructure.config import settings

DIMENSION = 512
EXPECTED_DISTANCE = Distance.COSINE
REQUIRED_PAYLOAD_INDEXES = {
    "face_id": PayloadSchemaType.KEYWORD,
    "active": PayloadSchemaType.BOOL,
    "model_version": PayloadSchemaType.KEYWORD,
}


class QdrantCollectionContractError(RuntimeError):
    """Raised when an existing collection does not match the required contract."""


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        url: str | None = None,
        collection_name: str | None = None,
        model_version: str | None = None,
    ) -> None:
        self._url = url or settings.qdrant_url
        self._collection_name = collection_name or settings.qdrant_collection_name
        self._model_version = model_version or settings.model_version
        self._client: AsyncQdrantClient | None = None
        self._ensure_lock = asyncio.Lock()
        self._validated = False

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

    def _validate_collection_contract(self, info: Any) -> None:
        vectors = info.config.params.vectors
        if not hasattr(vectors, "size") or vectors.size != DIMENSION:
            raise QdrantCollectionContractError(
                f"BLOCKED_QDRANT_COLLECTION_CONTRACT: dimension {getattr(vectors, 'size', None)} != {DIMENSION}"
            )
        if vectors.distance != EXPECTED_DISTANCE:
            raise QdrantCollectionContractError(
                f"BLOCKED_QDRANT_COLLECTION_CONTRACT: distance {vectors.distance} != {EXPECTED_DISTANCE}"
            )

        payload_schema = info.payload_schema or {}
        for field, expected_type in REQUIRED_PAYLOAD_INDEXES.items():
            index_info = payload_schema.get(field)
            if index_info is None:
                raise QdrantCollectionContractError(
                    f"BLOCKED_QDRANT_COLLECTION_CONTRACT: missing payload index {field}"
                )
            actual_type = getattr(index_info, "data_type", None)
            if actual_type != expected_type:
                raise QdrantCollectionContractError(
                    f"BLOCKED_QDRANT_COLLECTION_CONTRACT: {field} index type {actual_type} != {expected_type}"
                )

    def _raise_contract_error(self, reason: str) -> None:
        raise QdrantCollectionContractError(f"BLOCKED_QDRANT_COLLECTION_CONTRACT: {reason}")

    async def _ensure_collection(self) -> None:
        if self._validated:
            return

        async with self._ensure_lock:
            if self._validated:
                return

            collections = await self.client.get_collections()
            existing = any(c.name == self._collection_name for c in collections.collections)

            if existing:
                info = await self.client.get_collection(collection_name=self._collection_name)
                self._validate_collection_contract(info)
                self._validated = True
                return

            await self.client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE),
            )
            for field_name, field_schema in REQUIRED_PAYLOAD_INDEXES.items():
                await self.client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                    wait=True,
                )
            self._validated = True

    async def ensure_collection(self) -> None:
        """Public idempotent collection contract check (may create collection)."""
        await self._ensure_collection()

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
                    payload={
                        "sample_id": str(sample_id),
                        "face_id": str(face_id),
                        "active": True,
                        "model_version": self._model_version,
                    },
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
            query_filter=Filter(
                must=[
                    FieldCondition(key="active", match=MatchValue(value=True)),
                    FieldCondition(
                        key="model_version",
                        match=MatchValue(value=self._model_version),
                    ),
                ]
            ),
            limit=top_k,
            with_payload=True,
        )
        candidates: list[VectorCandidate] = []
        for point in result.points:
            payload = point.payload or {}
            face_id_str = payload.get("face_id")
            sample_id_str = payload.get("sample_id")
            point_model_version = payload.get("model_version")
            if face_id_str is None or sample_id_str is None:
                continue
            if str(point.id) != sample_id_str:
                continue
            if point_model_version != self._model_version:
                continue
            try:
                parsed_sample_id = uuid.UUID(str(point.id))
                parsed_face_id = uuid.UUID(face_id_str)
            except ValueError:
                continue
            score = point.score
            if not math.isfinite(score):
                continue
            candidates.append(
                VectorCandidate(
                    sample_id=SampleId(parsed_sample_id),
                    face_id=FaceId(parsed_face_id),
                    score=score,
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
