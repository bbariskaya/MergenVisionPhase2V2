"""Shared test fixtures."""

from __future__ import annotations

import os

import pytest

from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://mergenvision:mergenvision@localhost:5433/mergenvision",
)


@pytest.fixture
def unit_of_work() -> SqlAlchemyUnitOfWork:
    return SqlAlchemyUnitOfWork(async_session_maker)
