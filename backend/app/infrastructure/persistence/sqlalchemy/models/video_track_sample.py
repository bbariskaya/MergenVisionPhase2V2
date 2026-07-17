"""SQLAlchemy mapping for video_track_sample table."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    REAL,
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


class VideoTrackSampleOrm(Base):
    __tablename__ = "video_track_sample"

    __table_args__ = (
        UniqueConstraint("track_id", "sample_rank", name="uq_video_track_sample_track_rank"),
        CheckConstraint(
            "purpose IN ('identity_seed', 'best_shot', 'gallery_admission')",
            name="ck_video_track_sample_purpose",
        ),
        CheckConstraint("quality_score >= 0 AND quality_score <= 1", name="ck_video_track_sample_quality_range"),
        CheckConstraint("sample_rank >= 0", name="ck_video_track_sample_rank_nonnegative"),
    )

    track_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_track.track_id", ondelete="RESTRICT"), primary_key=True
    )
    sample_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("face_sample.sample_id", ondelete="RESTRICT"), primary_key=True
    )
    sample_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    quality_score: Mapped[float] = mapped_column(REAL, nullable=False)
    purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
