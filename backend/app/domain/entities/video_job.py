"""Video job domain entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from app.domain.value_objects import JobId, ProcessId, VideoId


@dataclass
class VideoJob:
    job_id: JobId
    video_id: VideoId
    process_id: ProcessId
    retry_of_job_id: JobId | None = None
    state: str = "pending"
    stage: str = "queued"
    progress_percent: int = 0
    sampling_mode: str = "every_frame"
    every_n_frames: int | None = None
    frames_per_second: Decimal | None = None
    processed_frames: int = 0
    sampled_frames: int = 0
    detected_observations: int = 0
    person_count: int = 0
    available_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    heartbeat_at: datetime | None = None
    attempt_count: int = 0
    max_attempts: int = 3
    cancellation_requested: bool = False
    error_code: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    cancelled_at: datetime | None = None
    version: int = 1
    result_manifest_bucket: str | None = None
    result_manifest_key: str | None = None
    result_manifest_sha256: str | None = None
    result_schema_version: str | None = None

    def request_cancellation(self) -> None:
        if self.state == "pending":
            self.state = "cancelled"
            self.cancelled_at = datetime.now(UTC)
        elif self.state == "processing":
            self.state = "cancelling"
            self.cancellation_requested = True
        # terminal states remain stable
        self.updated_at = datetime.now(UTC)
        self.version += 1

    def mark_cancelled(self) -> None:
        if self.state not in ("cancelling", "pending"):
            raise ValueError(f"Cannot mark job as cancelled from state {self.state}")
        self.state = "cancelled"
        self.cancelled_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)
        self.version += 1
