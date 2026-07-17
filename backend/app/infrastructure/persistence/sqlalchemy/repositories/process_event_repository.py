"""SQLAlchemy process_event repository."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities.process_event import ProcessEvent
from app.infrastructure.persistence.sqlalchemy.models.process_event import ProcessEventOrm


def _to_domain(orm: ProcessEventOrm) -> ProcessEvent:
    return ProcessEvent(
        event_id=orm.event_id,
        process_id=orm.process_id,
        job_id=orm.job_id,
        sequence_no=orm.sequence_no,
        event_type=orm.event_type,
        severity=orm.severity,
        payload=dict(orm.payload or {}),
        created_at=orm.created_at,
    )


class SqlAlchemyProcessEventRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, event: ProcessEvent) -> None:
        orm = ProcessEventOrm(
            event_id=event.event_id,
            process_id=event.process_id,
            job_id=event.job_id,
            sequence_no=event.sequence_no,
            event_type=event.event_type,
            severity=event.severity,
            payload=event.payload,
            created_at=event.created_at,
        )
        self._session.add(orm)

    async def next_sequence(self, process_id: str) -> int:
        result = await self._session.execute(
            select(ProcessEventOrm.sequence_no)
            .where(ProcessEventOrm.process_id == process_id)
            .order_by(ProcessEventOrm.sequence_no.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        return (row or 0) + 1

    async def list_by_process_id(self, process_id: str) -> list[ProcessEvent]:
        result = await self._session.execute(
            select(ProcessEventOrm)
            .where(ProcessEventOrm.process_id == process_id)
            .order_by(ProcessEventOrm.sequence_no)
        )
        return [_to_domain(orm) for orm in result.scalars().all()]
