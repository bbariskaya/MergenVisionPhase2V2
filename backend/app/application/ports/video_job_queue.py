"""Port for atomic video job queue operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from app.domain.entities.video_job import VideoJob
from app.domain.value_objects import JobId


@dataclass
class ClaimedVideoJob:
    job: VideoJob
    lease_token: UUID


class VideoJobQueue(Protocol):
    async def claim_next(
        self,
        worker_id: str,
        lease_token: UUID,
        now: datetime,
        lease_expires_at: datetime,
    ) -> ClaimedVideoJob | None: ...

    async def heartbeat(
        self,
        job_id: JobId,
        worker_id: str,
        lease_token: UUID,
        expected_version: int,
        now: datetime,
        new_lease_expires_at: datetime,
    ) -> bool: ...

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
    ) -> VideoJob | None: ...

    async def request_cancellation(self, job_id: JobId) -> VideoJob | None: ...

    async def mark_cancelled(
        self,
        job_id: JobId,
        worker_id: str,
        lease_token: UUID,
    ) -> VideoJob | None: ...

    async def mark_failed(
        self,
        job_id: JobId,
        worker_id: str,
        lease_token: UUID,
        error_code: str,
    ) -> VideoJob | None: ...

    async def release_for_retry(
        self,
        job_id: JobId,
        worker_id: str,
        lease_token: UUID,
        available_at: datetime,
        error_code: str,
    ) -> VideoJob | None: ...

    async def recover_expired_leases(
        self,
        now: datetime,
        batch_size: int,
    ) -> int: ...
