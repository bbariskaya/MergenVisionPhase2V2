"""SQLAlchemy process record repository."""

from __future__ import annotations

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities.process_record import ProcessRecord
from app.domain.value_objects import ProcessId
from app.infrastructure.persistence.sqlalchemy.models.process_record import ProcessRecordOrm


def _to_domain(orm: ProcessRecordOrm) -> ProcessRecord:
    return ProcessRecord(
        process_id=ProcessId(orm.process_id),
        process_type=orm.process_type,
        status=orm.status,
        face_count=orm.face_count,
        error_code=orm.error_code,
        details=dict(orm.details or {}),
        created_at=orm.created_at,
        completed_at=orm.completed_at,
    )


class SqlAlchemyProcessRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, process: ProcessRecord) -> None:
        orm = ProcessRecordOrm(
            process_id=process.process_id,
            process_type=process.process_type,
            status=process.status,
            face_count=process.face_count,
            error_code=process.error_code,
            details=process.details,
            created_at=process.created_at,
            completed_at=process.completed_at,
        )
        self._session.add(orm)

    async def get_by_id(self, process_id: ProcessId) -> ProcessRecord | None:
        result = await self._session.execute(
            select(ProcessRecordOrm).where(ProcessRecordOrm.process_id == process_id)
        )
        orm = result.scalar_one_or_none()
        return _to_domain(orm) if orm else None

    async def update(self, process: ProcessRecord) -> None:
        result = await self._session.execute(
            select(ProcessRecordOrm).where(ProcessRecordOrm.process_id == process.process_id)
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            raise ValueError(f"ProcessRecord {process.process_id} not found")
        orm.status = process.status
        orm.face_count = process.face_count
        orm.error_code = process.error_code
        orm.details = process.details
        orm.completed_at = process.completed_at

    async def list_by_status(self, status: str) -> list[ProcessRecord]:
        result = await self._session.execute(
            select(ProcessRecordOrm)
            .where(ProcessRecordOrm.status == status)
            .order_by(desc(ProcessRecordOrm.created_at))
        )
        return [_to_domain(orm) for orm in result.scalars().all()]
