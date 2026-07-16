"""SQLAlchemy async engine and session maker."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.infrastructure.config import settings

async_engine = create_async_engine(
    settings.database_url,
    future=True,
    echo=False,
    poolclass=NullPool,
)
async_session_maker = async_sessionmaker(async_engine, expire_on_commit=False)
