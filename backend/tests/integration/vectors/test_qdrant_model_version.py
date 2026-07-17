"""Phase 2 Milestone 0.6 — Qdrant model_version isolation tests."""

from __future__ import annotations

import pytest

from app.domain.value_objects import FaceId, SampleId
from app.infrastructure.uuid7 import generate_uuid7
from app.infrastructure.vectors.qdrant_adapter import QdrantVectorStore
from tests.fixtures.embedding_fixtures import vector_a

pytestmark = pytest.mark.asyncio


async def test_query_filters_by_model_version() -> None:
    """Only points with the adapter's model_version are returned."""
    collection_name = "mergenvision_s01_test_face_samples_v1"
    store_v1 = QdrantVectorStore(
        collection_name=collection_name,
        model_version="retinaface_r50_glintr100_v1",
    )
    store_v2 = QdrantVectorStore(
        collection_name=collection_name,
        model_version="retinaface_r50_glintr100_v2",
    )

    sample_v1 = SampleId(generate_uuid7())
    face_v1 = FaceId(generate_uuid7())
    sample_v2 = SampleId(generate_uuid7())
    face_v2 = FaceId(generate_uuid7())

    await store_v1.upsert(sample_v1, face_v1, vector_a())
    await store_v2.upsert(sample_v2, face_v2, vector_a())

    results_v1 = await store_v1.query(vector_a(), top_k=10)
    assert len(results_v1) == 1
    assert results_v1[0].sample_id == sample_v1
    assert results_v1[0].face_id == face_v1

    results_v2 = await store_v2.query(vector_a(), top_k=10)
    assert len(results_v2) == 1
    assert results_v2[0].sample_id == sample_v2
    assert results_v2[0].face_id == face_v2


async def test_existing_collection_contract_is_validated() -> None:
    """If the collection exists but has wrong contract, validation raises."""
    import qdrant_client.models as qmodels

    collection_name = "mergenvision_test_bad_contract"
    client = QdrantVectorStore().client
    try:
        if any(c.name == collection_name for c in (await client.get_collections()).collections):
            await client.delete_collection(collection_name)

        await client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=512, distance=qmodels.Distance.COSINE),
        )
        # Missing required payload indexes intentionally.

        bad_store = QdrantVectorStore(
            collection_name=collection_name,
            model_version="retinaface_r50_glintr100_v1",
        )
        with pytest.raises(RuntimeError, match="BLOCKED_QDRANT_COLLECTION_CONTRACT"):
            await bad_store.query(vector_a(), top_k=1)
    finally:
        await client.delete_collection(collection_name)
