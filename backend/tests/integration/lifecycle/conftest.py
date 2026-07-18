"""Lifecycle integration test configuration and fixtures."""

from __future__ import annotations

import pytest

from tests.support.resource_guard import assert_safe_test_environment

assert_safe_test_environment()

from app.application.ports.id_generator import IdGenerator
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.infrastructure.storage.minio_adapter import MinIOObjectStore
from app.infrastructure.vectors.qdrant_adapter import QdrantVectorStore

pytestmark = pytest.mark.asyncio(scope="session")


@pytest.fixture
def lifecycle_service(
    unit_of_work_factory: UnitOfWorkFactory,
    object_store: MinIOObjectStore,
    vector_store: QdrantVectorStore,
    id_generator: IdGenerator,
) -> IdentityStorageLifecycleService:
    return IdentityStorageLifecycleService(
        unit_of_work_factory=unit_of_work_factory,
        object_store=object_store,
        vector_store=vector_store,
        id_generator=id_generator,
    )
