"""Atomic video job queue implementation using SQLAlchemy."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.ports.video_job_queue import ClaimedVideoJob, VideoJobQueue
from app.domain.entities.video_job import VideoJob
from app.domain.value_objects import JobId
from app.infrastructure.persistence.sqlalchemy.models.video_job import VideoJobOrm
from app.infrastructure.persistence.sqlalchemy.repositories.video_repositories import (
    _job_to_domain,
)


class SqlAlchemyVideoJobQueue(VideoJobQueue):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def claim_next(
        self,
        worker_id: str,
        lease_token: UUID,
        now: datetime,
        lease_expires_at: datetime,
    ) -> ClaimedVideoJob | None:
        result = await self._session.execute(
            select(VideoJobOrm)
            .where(
                VideoJobOrm.state == "pending",
                VideoJobOrm.available_at <= now,
            )
            .order_by(VideoJobOrm.available_at, VideoJobOrm.created_at, VideoJobOrm.job_id)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            return None

        orm.state = "processing"
        orm.stage = "download"
        orm.lease_owner = worker_id
        orm.lease_token = lease_token
        orm.lease_expires_at = lease_expires_at
        orm.heartbeat_at = now
        orm.attempt_count += 1
        orm.updated_at = datetime.now(UTC)
        orm.version += 1
        if orm.started_at is None:
            orm.started_at = now

        return ClaimedVideoJob(job=_job_to_domain(orm), lease_token=lease_token)

    async def heartbeat(
        self,
        job_id: JobId,
        worker_id: str,
        lease_token: UUID,
        expected_version: int,
        now: datetime,
        new_lease_expires_at: datetime,
    ) -> bool:
        result = await self._session.execute(
            select(VideoJobOrm)
            .where(
                VideoJobOrm.job_id == job_id,
                VideoJobOrm.lease_owner == worker_id,
                VideoJobOrm.lease_token == lease_token,
                VideoJobOrm.version == expected_version,
            )
            .with_for_update()
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            return False
        orm.heartbeat_at = now
        orm.lease_expires_at = new_lease_expires_at
        orm.updated_at = datetime.now(UTC)
        orm.version += 1
        return True

    async def update_stage(
        self,
        job_id: JobId,
        worker_id: str,
        lease_token: UUID,
        expected_version: int,
        *,
        stage: str,
        progress_percent: int,
        processed_frames: int,
        sampled_frames: int,
        detected_observations: int,
        person_count: int,
    ) -> VideoJob | None:
        result = await self._session.execute(
            select(VideoJobOrm)
            .where(
                VideoJobOrm.job_id == job_id,
                VideoJobOrm.lease_owner == worker_id,
                VideoJobOrm.lease_token == lease_token,
                VideoJobOrm.version == expected_version,
            )
            .with_for_update()
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            return None
        orm.stage = stage
        orm.progress_percent = progress_percent
        orm.processed_frames = processed_frames
        orm.sampled_frames = sampled_frames
        orm.detected_observations = detected_observations
        orm.person_count = person_count
        orm.updated_at = datetime.now(UTC)
        orm.version += 1
        return _job_to_domain(orm)

    async def request_cancellation(self, job_id: JobId) -> VideoJob | None:
        result = await self._session.execute(
            select(VideoJobOrm)
            .where(VideoJobOrm.job_id == job_id)
            .with_for_update()
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            return None
        if orm.state == "pending":
            orm.state = "cancelled"
            orm.cancelled_at = datetime.now(UTC)
        elif orm.state == "processing":
            orm.state = "cancelling"
            orm.cancellation_requested = True
        elif orm.state == "cancelling":
            orm.cancellation_requested = True
        else:
            return _job_to_domain(orm)
        orm.updated_at = datetime.now(UTC)
        orm.version += 1
        return _job_to_domain(orm)

    async def mark_cancelled(
        self,
        job_id: JobId,
        worker_id: str,
        lease_token: UUID,
    ) -> VideoJob | None:
        result = await self._session.execute(
            select(VideoJobOrm)
            .where(
                VideoJobOrm.job_id == job_id,
                VideoJobOrm.lease_owner == worker_id,
                VideoJobOrm.lease_token == lease_token,
                VideoJobOrm.state.in_(["cancelling", "pending"]),
            )
            .with_for_update()
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            return None
        orm.state = "cancelled"
        orm.cancelled_at = datetime.now(UTC)
        orm.lease_owner = None
        orm.lease_token = None
        orm.lease_expires_at = None
        orm.heartbeat_at = None
        orm.updated_at = datetime.now(UTC)
        orm.version += 1
        return _job_to_domain(orm)

    async def mark_failed(
        self,
        job_id: JobId,
        worker_id: str,
        lease_token: UUID,
        error_code: str,
    ) -> VideoJob | None:
        result = await self._session.execute(
            select(VideoJobOrm)
            .where(
                VideoJobOrm.job_id == job_id,
                VideoJobOrm.lease_owner == worker_id,
                VideoJobOrm.lease_token == lease_token,
            )
            .with_for_update()
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            return None
        orm.state = "failed"
        orm.error_code = error_code
        orm.failed_at = datetime.now(UTC)
        orm.lease_owner = None
        orm.lease_token = None
        orm.lease_expires_at = None
        orm.heartbeat_at = None
        orm.updated_at = datetime.now(UTC)
        orm.version += 1
        return _job_to_domain(orm)

    async def release_for_retry(
        self,
        job_id: JobId,
        worker_id: str,
        lease_token: UUID,
        available_at: datetime,
        error_code: str,
    ) -> VideoJob | None:
        result = await self._session.execute(
            select(VideoJobOrm)
            .where(
                VideoJobOrm.job_id == job_id,
                VideoJobOrm.lease_owner == worker_id,
                VideoJobOrm.lease_token == lease_token,
                VideoJobOrm.state == "processing",
            )
            .with_for_update()
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            return None
        orm.state = "pending"
        orm.stage = "queued"
        orm.available_at = available_at
        orm.error_code = error_code
        orm.lease_owner = None
        orm.lease_token = None
        orm.lease_expires_at = None
        orm.heartbeat_at = None
        orm.updated_at = datetime.now(UTC)
        orm.version += 1
        return _job_to_domain(orm)

    async def recover_expired_leases(
        self,
        now: datetime,
        batch_size: int,
    ) -> int:
        rows = await self._session.execute(
            select(VideoJobOrm)
            .where(
                VideoJobOrm.state.in_(["processing", "cancelling"]),
                VideoJobOrm.lease_expires_at < now,
            )
            .limit(batch_size)
            .with_for_update(skip_locked=True)
        )
        count = 0
        for orm in rows.scalars().all():
            if orm.state == "cancelling" or orm.cancellation_requested:
                orm.state = "cancelled"
                orm.cancelled_at = datetime.now(UTC)
            elif orm.attempt_count >= orm.max_attempts:
                orm.state = "failed"
                orm.error_code = "max_retry_exceeded"
                orm.failed_at = datetime.now(UTC)
            else:
                orm.state = "pending"
                orm.stage = "queued"
                orm.available_at = now
                orm.error_code = "lease_expired"
            orm.lease_owner = None
            orm.lease_token = None
            orm.lease_expires_at = None
            orm.heartbeat_at = None
            orm.updated_at = datetime.now(UTC)
            orm.version += 1
            count += 1
        return count
