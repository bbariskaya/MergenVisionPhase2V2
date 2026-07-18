"""Qdrant store writing into Phase 2's existing collection.

Point contract (mirrors ``QdrantVectorStore``):
- ``point_id`` = ``sample_id`` (UUID string)
- vector = 512-dim float embedding
- payload = ``{"sample_id": ..., "face_id": ..., "active": true, "model_version": ...}``
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

DIMENSION = 512
EXPECTED_DISTANCE = Distance.COSINE
REQUIRED_PAYLOAD_INDEXES = {
    "face_id": PayloadSchemaType.KEYWORD,
    "active": PayloadSchemaType.BOOL,
    "model_version": PayloadSchemaType.KEYWORD,
}


class QdrantCollectionContractError(RuntimeError):
    """Raised when an existing collection does not match the required contract."""


class QdrantStore:
    """Bounded-concurrent Qdrant upsert store for face embeddings."""

    def __init__(
        self,
        url: str,
        collection_name: str,
        model_version: str,
        vector_size: int = DIMENSION,
        distance: Distance = EXPECTED_DISTANCE,
        max_concurrency: int = 32,
    ) -> None:
        self._client = QdrantClient(url=url)
        self._collection_name = collection_name
        self._model_version = model_version
        self._vector_size = vector_size
        self._distance = distance
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._validated = False

    @staticmethod
    def _validate_embedding(embedding: Sequence[float]) -> None:
        if len(embedding) != DIMENSION:
            raise ValueError(f"embedding dimension {len(embedding)} != {DIMENSION}")
        if not all(math.isfinite(v) for v in embedding):
            raise ValueError("embedding values must be finite")
        norm = math.sqrt(sum(v * v for v in embedding))
        if norm == 0.0:
            raise ValueError("embedding norm must be non-zero")

    def _validate_collection_contract(self, info: Any) -> None:
        vectors = info.config.params.vectors
        if not hasattr(vectors, "size") or vectors.size != self._vector_size:
            raise QdrantCollectionContractError(f"dimension {getattr(vectors, 'size', None)} != {self._vector_size}")
        if vectors.distance != self._distance:
            raise QdrantCollectionContractError(f"distance {vectors.distance} != {self._distance}")
        payload_schema = info.payload_schema or {}
        for field, expected_type in REQUIRED_PAYLOAD_INDEXES.items():
            index_info = payload_schema.get(field)
            if index_info is None:
                raise QdrantCollectionContractError(f"missing payload index {field}")
            actual_type = getattr(index_info, "data_type", None)
            if actual_type != expected_type:
                raise QdrantCollectionContractError(f"{field} index type {actual_type} != {expected_type}")

    def _ensure_collection_sync(self) -> None:
        """Validate contract; create collection only if it does not exist.

        Any exception other than "collection not found" is treated as a
        fail-closed contract/network/auth error.
        """
        if self._validated:
            return
        try:
            info = self._client.get_collection(self._collection_name)
        except Exception as exc:
            message = str(exc).lower()
            if "not found" in message or "doesn't exist" in message or "does not exist" in message:
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=self._vector_size, distance=self._distance),
                )
                for field_name, field_schema in REQUIRED_PAYLOAD_INDEXES.items():
                    self._client.create_payload_index(
                        collection_name=self._collection_name,
                        field_name=field_name,
                        field_schema=field_schema,
                        wait=True,
                    )
                self._validated = True
                return
            raise QdrantCollectionContractError(f"BLOCKED_QDRANT_CONTRACT: cannot validate collection: {exc}") from exc

        self._validate_collection_contract(info)
        self._validated = True

    async def ensure_collection(self) -> None:
        await asyncio.to_thread(self._ensure_collection_sync)

    def _upsert_sync(
        self,
        sample_id: str,
        face_id: str,
        embedding: np.ndarray | Sequence[float],
        active: bool = True,
        batch: bool = False,
    ) -> Any:
        vector = np.asarray(embedding, dtype=np.float32).flatten().tolist()
        self._validate_embedding(vector)
        point = PointStruct(
            id=sample_id,
            vector=vector,
            payload={
                "sample_id": sample_id,
                "face_id": face_id,
                "active": active,
                "model_version": self._model_version,
            },
        )
        if batch:
            return point
        return self._client.upsert(
            collection_name=self._collection_name,
            points=[point],
            wait=True,
        )

    async def upsert(
        self,
        sample_id: str,
        face_id: str,
        embedding: np.ndarray | Sequence[float],
        active: bool = True,
    ) -> Any:
        await self.ensure_collection()
        async with self._semaphore:
            return await asyncio.to_thread(
                self._upsert_sync,
                sample_id,
                face_id,
                embedding,
                active,
                False,
            )

    async def upsert_many(
        self,
        items: Sequence[tuple[str, str, np.ndarray | Sequence[float]]],
        active: bool = True,
    ) -> Any:
        """Batch upsert with bounded concurrency.

        ``items`` are tuples of ``(sample_id, face_id, embedding)``.
        """
        await self.ensure_collection()
        points: list[PointStruct] = []
        for sample_id, face_id, embedding in items:
            points.append(self._upsert_sync(sample_id, face_id, embedding, active, batch=True))
        async with self._semaphore:
            return await asyncio.to_thread(
                self._client.upsert,
                collection_name=self._collection_name,
                points=points,
                wait=True,
            )

    async def set_active(self, sample_id: str, active: bool) -> Any:
        await self.ensure_collection()
        async with self._semaphore:
            return await asyncio.to_thread(
                self._client.set_payload,
                collection_name=self._collection_name,
                payload={"active": active},
                points=[sample_id],
            )

    def _retrieve_sync(self, sample_id: str) -> Any:
        records = self._client.retrieve(
            collection_name=self._collection_name,
            ids=[sample_id],
            with_payload=True,
            with_vectors=False,
        )
        return records[0] if records else None

    async def retrieve(self, sample_id: str) -> Any:
        """Fetch a single point by sample id, returning ``None`` if absent."""
        await self.ensure_collection()
        async with self._semaphore:
            return await asyncio.to_thread(self._retrieve_sync, sample_id)

    async def delete_best_effort(self, sample_id: str) -> None:
        try:
            await self.ensure_collection()
            async with self._semaphore:
                await asyncio.to_thread(
                    self._client.delete,
                    collection_name=self._collection_name,
                    points_selector=[sample_id],
                )
        except Exception:
            pass
