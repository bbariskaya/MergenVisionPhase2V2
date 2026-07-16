"""Integration tests for Qdrant vector-store adapter."""

from __future__ import annotations

import pytest

from app.domain.errors import ValidationError
from app.domain.value_objects import FaceId, SampleId
from app.infrastructure.uuid7 import generate_uuid7
from app.infrastructure.vectors.qdrant_adapter import QdrantVectorStore
from tests.fixtures.embedding_fixtures import (
    cosine_similarity,
    vector_a,
    vector_b,
)


async def test_upsert_and_query_vector_a() -> None:
    store = QdrantVectorStore()
    sample_id = SampleId(generate_uuid7())
    face_id = FaceId(generate_uuid7())

    await store.upsert(sample_id, face_id, vector_a())

    results = await store.query(vector_a(), top_k=1)
    assert len(results) == 1
    assert results[0].sample_id == sample_id
    assert results[0].face_id == face_id
    assert results[0].score > 0.99


async def test_orthogonal_vector_returns_near_zero_score() -> None:
    store = QdrantVectorStore()
    sample_id = SampleId(generate_uuid7())
    face_id = FaceId(generate_uuid7())

    await store.upsert(sample_id, face_id, vector_a())

    results = await store.query(vector_b(), top_k=1)
    assert len(results) == 1
    assert results[0].sample_id == sample_id
    assert results[0].face_id == face_id
    assert results[0].score < 0.01


async def test_active_filter_excludes_inactive() -> None:
    store = QdrantVectorStore()
    sample_id = SampleId(generate_uuid7())
    face_id = FaceId(generate_uuid7())

    await store.upsert(sample_id, face_id, vector_a())
    await store.set_active(sample_id, False)

    results = await store.query(vector_a(), top_k=1)
    assert len(results) == 0


async def test_vector_must_be_unit_normalized_length_512() -> None:
    store = QdrantVectorStore()

    with pytest.raises(ValidationError):
        await store.upsert(SampleId(generate_uuid7()), FaceId(generate_uuid7()), [1.0, 0.0])

    with pytest.raises(ValidationError):
        await store.upsert(
            SampleId(generate_uuid7()), FaceId(generate_uuid7()), [float("inf"), *([0.0] * 510)]
        )


async def test_cosine_of_fixtures_is_zero() -> None:
    assert abs(cosine_similarity(vector_a(), vector_b())) < 1e-9
