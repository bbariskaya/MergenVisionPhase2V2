"""SQLAlchemy mapping for video_track table."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    REAL,
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.sqlalchemy.base import Base


class VideoTrackOrm(Base):
    __tablename__ = "video_track"

    __table_args__ = (
        UniqueConstraint("job_id", "track_ordinal", name="uq_video_track_job_ordinal"),
        CheckConstraint(
            "status_at_processing IN ('known', 'anonymous', 'new_anonymous')",
            name="ck_video_track_status",
        ),
        CheckConstraint("first_frame_index <= last_frame_index", name="ck_video_track_frame_order"),
        CheckConstraint("first_pts_ns <= last_pts_ns", name="ck_video_track_pts_order"),
        CheckConstraint("total_duration_ns >= 0", name="ck_video_track_duration_nonnegative"),
        CheckConstraint("detection_count >= 0", name="ck_video_track_detection_count_nonnegative"),
        CheckConstraint("tracklet_count >= 0", name="ck_video_track_tracklet_count_nonnegative"),
        CheckConstraint("match_confidence >= 0 AND match_confidence <= 1", name="ck_video_track_confidence_range"),
        CheckConstraint(
            "(status_at_processing = 'known' AND name_at_processing IS NOT NULL AND btrim(name_at_processing) != '') "
            "OR (status_at_processing IN ('anonymous', 'new_anonymous') AND name_at_processing IS NULL)",
            name="ck_video_track_status_name_consistency",
        ),
    )

    track_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_job.job_id", ondelete="RESTRICT"), nullable=False
    )
    track_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    face_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("face_identity.face_id", ondelete="RESTRICT"), nullable=False
    )
    recognition_result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("recognition_result.result_id", ondelete="RESTRICT"), nullable=False, unique=True
    )
    status_at_processing: Mapped[str] = mapped_column(String(16), nullable=False)
    name_at_processing: Mapped[str | None] = mapped_column(String(255), nullable=True)
    metadata_at_processing: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}::jsonb")
    identity_version_at_processing: Mapped[int] = mapped_column(Integer, nullable=False)
    match_confidence: Mapped[float] = mapped_column(REAL, nullable=False)
    top1_score: Mapped[float | None] = mapped_column(REAL, nullable=True)
    top2_score: Mapped[float | None] = mapped_column(REAL, nullable=True)
    margin_score: Mapped[float | None] = mapped_column(REAL, nullable=True)
    threshold_used: Mapped[float | None] = mapped_column(REAL, nullable=True)
    first_frame_index: Mapped[int] = mapped_column(BigInteger, nullable=False)
    last_frame_index: Mapped[int] = mapped_column(BigInteger, nullable=False)
    first_pts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    last_pts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    total_duration_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    detection_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    tracklet_count: Mapped[int] = mapped_column(Integer, nullable=False)
    best_sample_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("face_sample.sample_id", ondelete="RESTRICT"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
