"""SQLAlchemy mapping for video_timeline_chunk table."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.sqlalchemy.base import Base


class VideoTimelineChunkOrm(Base):
    __tablename__ = "video_timeline_chunk"

    __table_args__ = (
        UniqueConstraint("job_id", "artifact_kind", "sequence_no", name="uq_video_timeline_chunk_job_kind_seq"),
        UniqueConstraint("bucket", "object_key", name="uq_video_timeline_chunk_bucket_key"),
        CheckConstraint(
            "artifact_kind IN ('private_observation', 'public_overlay')",
            name="ck_video_timeline_chunk_artifact_kind",
        ),
        CheckConstraint("size_bytes >= 0", name="ck_video_timeline_chunk_size_nonnegative"),
        CheckConstraint("record_count >= 0", name="ck_video_timeline_chunk_record_count_nonnegative"),
    )

    chunk_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_job.job_id", ondelete="RESTRICT"), nullable=False
    )
    artifact_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False)
    start_pts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_pts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    record_count: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    compression: Mapped[str] = mapped_column(String(32), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
