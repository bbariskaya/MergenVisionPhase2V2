"""Vectors integration test configuration."""

from __future__ import annotations

import pytest

from tests.support.resource_guard import assert_safe_test_environment

assert_safe_test_environment()

from app.infrastructure.vectors.qdrant_adapter import QdrantVectorStore

pytestmark = pytest.mark.asyncio(scope="session")


@pytest.fixture(autouse=True)
async def _clean_qdrant_collection(vector_store: QdrantVectorStore) -> None:
    collections = await vector_store.client.get_collections()
    if any(c.name == vector_store._collection_name for c in collections.collections):
        await vector_store.client.delete_collection(collection_name=vector_store._collection_name)
