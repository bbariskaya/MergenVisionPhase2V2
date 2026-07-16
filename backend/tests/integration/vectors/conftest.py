"""Vectors integration test configuration."""

from collections.abc import AsyncGenerator

import pytest

pytestmark = pytest.mark.asyncio(scope="session")


@pytest.fixture(autouse=True)
async def _clean_qdrant_collection() -> AsyncGenerator[None, None]:
    from app.infrastructure.vectors.qdrant_adapter import QdrantVectorStore

    store = QdrantVectorStore()
    try:
        collections = await store.client.get_collections()
        if any(c.name == store._collection_name for c in collections.collections):
            await store.client.delete_collection(collection_name=store._collection_name)
    except Exception:
        pass
    yield
