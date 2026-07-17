"""SQLAlchemy outbox_event repository."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities.outbox_event import OutboxEvent
from app.infrastructure.persistence.sqlalchemy.models.outbox_event import OutboxEventOrm


def _to_domain(orm: OutboxEventOrm) -> OutboxEvent:
    return OutboxEvent(
        outbox_event_id=orm.outbox_event_id,
        aggregate_type=orm.aggregate_type,
        aggregate_id=orm.aggregate_id,
        event_type=orm.event_type,
        dedupe_key=orm.dedupe_key,
        state=orm.state,
        attempt_count=orm.attempt_count,
        max_attempts=orm.max_attempts,
        available_at=orm.available_at,
        locked_by=orm.locked_by,
        locked_until=orm.locked_until,
        payload=dict(orm.payload or {}),
        last_error_code=orm.last_error_code,
        created_at=orm.created_at,
        updated_at=orm.updated_at,
        succeeded_at=orm.succeeded_at,
        failed_at=orm.failed_at,
    )


class SqlAlchemyOutboxEventRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, event: OutboxEvent) -> None:
        orm = OutboxEventOrm(
            outbox_event_id=event.outbox_event_id,
            aggregate_type=event.aggregate_type,
            aggregate_id=event.aggregate_id,
            event_type=event.event_type,
            dedupe_key=event.dedupe_key,
            state=event.state,
            attempt_count=event.attempt_count,
            max_attempts=event.max_attempts,
            available_at=event.available_at,
            locked_by=event.locked_by,
            locked_until=event.locked_until,
            payload=event.payload,
            last_error_code=event.last_error_code,
            created_at=event.created_at,
            updated_at=event.updated_at,
            succeeded_at=event.succeeded_at,
            failed_at=event.failed_at,
        )
        self._session.add(orm)

    async def get_by_dedupe_key(self, dedupe_key: str) -> OutboxEvent | None:
        result = await self._session.execute(
            select(OutboxEventOrm).where(OutboxEventOrm.dedupe_key == dedupe_key)
        )
        orm = result.scalar_one_or_none()
        return _to_domain(orm) if orm else None

    async def get_by_id(self, outbox_event_id: UUID) -> OutboxEvent | None:
        result = await self._session.execute(
            select(OutboxEventOrm).where(OutboxEventOrm.outbox_event_id == outbox_event_id)
        )
        orm = result.scalar_one_or_none()
        return _to_domain(orm) if orm else None

    async def update(self, event: OutboxEvent) -> None:
        result = await self._session.execute(
            select(OutboxEventOrm).where(
                OutboxEventOrm.outbox_event_id == event.outbox_event_id
            )
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            raise ValueError(f"OutboxEvent {event.outbox_event_id} not found")
        orm.state = event.state
        orm.attempt_count = event.attempt_count
        orm.locked_by = event.locked_by
        orm.locked_until = event.locked_until
        orm.last_error_code = event.last_error_code
        orm.succeeded_at = event.succeeded_at
        orm.failed_at = event.failed_at
        orm.updated_at = event.updated_at
