"""Shared integration test configuration.

This module is loaded before any backend/tests/integration/*/conftest.py. It
validates the dedicated test environment and provides shared infrastructure
fixtures.
"""

from __future__ import annotations

import pytest

from tests.support.resource_guard import assert_safe_test_environment

# Fail-fast if the process is not configured for the isolated test namespace.
assert_safe_test_environment()

from app.application.ports.id_generator import IdGenerator
from app.application.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from app.infrastructure.storage.minio_adapter import MinIOObjectStore
from app.infrastructure.vectors.qdrant_adapter import QdrantVectorStore


@pytest.fixture
def unit_of_work() -> UnitOfWork:
    return SqlAlchemyUnitOfWork(async_session_maker)


@pytest.fixture
def unit_of_work_factory() -> UnitOfWorkFactory:
    def _factory() -> UnitOfWork:
        return SqlAlchemyUnitOfWork(async_session_maker)

    return _factory


@pytest.fixture
def object_store() -> MinIOObjectStore:
    return MinIOObjectStore()


@pytest.fixture
def vector_store() -> QdrantVectorStore:
    return QdrantVectorStore()


@pytest.fixture
def id_generator() -> IdGenerator:
    from app.infrastructure.uuid7 import Uuid7Generator

    return Uuid7Generator()
