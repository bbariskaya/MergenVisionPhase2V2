"""SQLAlchemy async Unit of Work."""

from __future__ import annotations

from types import TracebackType
from typing import Self

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.application.ports.unit_of_work import UnitOfWork
from app.infrastructure.persistence.sqlalchemy.repositories import (
    SqlAlchemyFaceIdentityRepository,
    SqlAlchemyFaceSampleRepository,
    SqlAlchemyProcessRepository,
    SqlAlchemyRecognitionResultRepository,
)


class SqlAlchemyUnitOfWork(UnitOfWork):
    def __init__(self, session_maker: async_sessionmaker[AsyncSession]) -> None:
        self._session_maker = session_maker
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> Self:
        self._session = self._session_maker()
        await self._session.begin()
        self.face_identities = SqlAlchemyFaceIdentityRepository(self._session)  # type: ignore[assignment]
        self.face_samples = SqlAlchemyFaceSampleRepository(self._session)  # type: ignore[assignment]
        self.processes = SqlAlchemyProcessRepository(self._session)  # type: ignore[assignment]
        self.recognition_results = SqlAlchemyRecognitionResultRepository(self._session)  # type: ignore[assignment]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._session is None:
            return
        try:
            if exc_type is not None:
                await self._session.rollback()
            await self._session.close()
        finally:
            self._session = None

    async def commit(self) -> None:
        if self._session is None:
            raise RuntimeError("Unit of work is not active")
        await self._session.commit()

    async def rollback(self) -> None:
        if self._session is None:
            raise RuntimeError("Unit of work is not active")
        await self._session.rollback()
