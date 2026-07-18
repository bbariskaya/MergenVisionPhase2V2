"""Qdrant store writing into Phase 2's existing ``faces`` collection.

Point contract (mirrors ``QdrantAdapter``):
- ``point_id`` = ``sample_id`` (UUID string)
- vector = 512-dim float embedding
- payload = ``{"sample_id": ..., "face_id": ..., "active": true, "model_version": ...}``
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantStore:
    """Bounded-concurrent Qdrant upsert store for face embeddings."""

    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int = 512,
        distance: Distance = Distance.COSINE,
        max_concurrency: int = 32,
    ) -> None:
        self._client = QdrantClient(url=url)
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._distance = distance
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def _ensure_collection_sync(self) -> None:
        """Create collection only if it does not exist (idempotent)."""
        try:
            self._client.get_collection(self._collection_name)
        except Exception:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=self._vector_size, distance=self._distance),
            )

    async def ensure_collection(self) -> None:
        await asyncio.to_thread(self._ensure_collection_sync)

    def _upsert_sync(
        self,
        sample_id: str,
        face_id: str,
        embedding: np.ndarray | Sequence[float],
        model_version: str,
        active: bool = True,
        batch: bool = False,
    ) -> Any:
        vector = np.asarray(embedding, dtype=np.float32).flatten().tolist()
        if len(vector) != self._vector_size:
            raise ValueError(
                f"embedding dimension {len(vector)} does not match collection size {self._vector_size}"
            )
        point = PointStruct(
            id=sample_id,
            vector=vector,
            payload={
                "sample_id": sample_id,
                "face_id": face_id,
                "active": active,
                "model_version": model_version,
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
        model_version: str,
        active: bool = True,
    ) -> Any:
        await self.ensure_collection()
        async with self._semaphore:
            return await asyncio.to_thread(
                self._upsert_sync,
                sample_id,
                face_id,
                embedding,
                model_version,
                active,
                False,
            )

    async def upsert_many(
        self,
        items: Sequence[tuple[str, str, np.ndarray | Sequence[float], str]],
        active: bool = True,
    ) -> Any:
        """Batch upsert with bounded concurrency.

        ``items`` are tuples of ``(sample_id, face_id, embedding, model_version)``.
        """
        await self.ensure_collection()
        points: list[PointStruct] = []
        for sample_id, face_id, embedding, model_version in items:
            points.append(
                self._upsert_sync(
                    sample_id, face_id, embedding, model_version, active, batch=True
                )
            )
        async with self._semaphore:
            return await asyncio.to_thread(
                self._client.upsert,
                collection_name=self._collection_name,
                points=points,
                wait=True,
            )

    async def set_active(self, sample_id: str, active: bool) -> Any:
        async with self._semaphore:
            return await asyncio.to_thread(
                self._client.set_payload,
                collection_name=self._collection_name,
                payload={"active": active},
                points=[sample_id],
            )

    async def delete_best_effort(self, sample_id: str) -> None:
        try:
            async with self._semaphore:
                await asyncio.to_thread(
                    self._client.delete,
                    collection_name=self._collection_name,
                    points_selector=[sample_id],
                )
        except Exception:
            pass
