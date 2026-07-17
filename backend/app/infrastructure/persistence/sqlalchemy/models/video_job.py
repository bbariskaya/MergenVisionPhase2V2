"""SQLAlchemy mapping for video_job table."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.sqlalchemy.base import Base


class VideoJobOrm(Base):
    __tablename__ = "video_job"

    __table_args__ = (
        CheckConstraint(
            "state IN ('pending', 'processing', 'cancelling', 'completed', 'failed', 'cancelled')",
            name="ck_video_job_state",
        ),
        CheckConstraint(
            "stage IN ('queued', 'download', 'probe', 'decode_infer', 'track_reconcile', 'persist', 'finalize', 'cleanup')",
            name="ck_video_job_stage",
        ),
        CheckConstraint(
            "progress_percent >= 0 AND progress_percent <= 100",
            name="ck_video_job_progress_range",
        ),
        CheckConstraint("processed_frames >= 0", name="ck_video_job_processed_frames_nonneg"),
        CheckConstraint("sampled_frames >= 0", name="ck_video_job_sampled_frames_nonneg"),
        CheckConstraint("detected_observations >= 0", name="ck_video_job_observations_nonneg"),
        CheckConstraint("person_count >= 0", name="ck_video_job_person_count_nonneg"),
        CheckConstraint("attempt_count >= 0", name="ck_video_job_attempt_count_nonneg"),
        CheckConstraint("max_attempts >= 1", name="ck_video_job_max_attempts_positive"),
        CheckConstraint(
            """
            (sampling_mode = 'every_frame' AND every_n_frames IS NULL AND frames_per_second IS NULL)
            OR
            (sampling_mode = 'every_n_frames' AND every_n_frames IS NOT NULL AND every_n_frames > 0 AND frames_per_second IS NULL)
            OR
            (sampling_mode = 'frames_per_second' AND frames_per_second IS NOT NULL AND frames_per_second > 0 AND every_n_frames IS NULL)
            """,
            name="ck_video_job_sampling_mode",
        ),
        CheckConstraint(
            "state != 'completed' OR (progress_percent = 100 AND completed_at IS NOT NULL)",
            name="ck_video_job_completed_requires_progress",
        ),
        CheckConstraint(
            "state != 'failed' OR (failed_at IS NOT NULL AND error_code IS NOT NULL AND btrim(error_code) != '')",
            name="ck_video_job_failed_requires_fields",
        ),
        CheckConstraint(
            "state != 'cancelled' OR cancelled_at IS NOT NULL",
            name="ck_video_job_cancelled_requires_timestamp",
        ),
        CheckConstraint(
            """
            state NOT IN ('processing', 'cancelling')
            OR (lease_owner IS NOT NULL AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL AND heartbeat_at IS NOT NULL)
            """,
            name="ck_video_job_processing_requires_lease",
        ),
        CheckConstraint(
            """
            state IN ('processing', 'cancelling')
            OR (lease_owner IS NULL AND lease_token IS NULL AND lease_expires_at IS NULL AND heartbeat_at IS NULL)
            """,
            name="ck_video_job_non_processing_no_lease",
        ),
        CheckConstraint(
            """
            state != 'completed' OR (
                result_manifest_bucket IS NOT NULL AND btrim(result_manifest_bucket) != ''
                AND result_manifest_key IS NOT NULL AND btrim(result_manifest_key) != ''
                AND result_manifest_sha256 IS NOT NULL AND btrim(result_manifest_sha256) != ''
                AND result_schema_version IS NOT NULL AND btrim(result_schema_version) != ''
            )
            """,
            name="ck_video_job_completed_requires_manifest",
        ),
    )

    job_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_asset.video_id", ondelete="RESTRICT"), nullable=False
    )
    process_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("process_record.process_id", ondelete="RESTRICT"), nullable=False, unique=True
    )
    retry_of_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_job.job_id", ondelete="RESTRICT"), nullable=True
    )
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    progress_percent: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("0")
    )
    sampling_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    every_n_frames: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frames_per_second: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    processed_frames: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0")
    )
    sampled_frames: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0")
    )
    detected_observations: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0")
    )
    person_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    lease_owner: Mapped[str | None] = mapped_column(String(255), nullable=True)
    lease_token: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    cancellation_requested: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))

    result_manifest_bucket: Mapped[str | None] = mapped_column(String(255), nullable=True)
    result_manifest_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    result_manifest_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    result_schema_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
