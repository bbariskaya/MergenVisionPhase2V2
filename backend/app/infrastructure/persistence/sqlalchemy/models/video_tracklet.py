"""SQLAlchemy mapping for video_tracklet table."""

from __future__ import annotations

import uuid
from datetime import datetime

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
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.sqlalchemy.base import Base


class VideoTrackletOrm(Base):
    __tablename__ = "video_tracklet"

    __table_args__ = (
        UniqueConstraint("job_id", "track_id", "tracklet_ordinal", name="uq_video_tracklet_job_ordinal"),
        CheckConstraint(
            "state IN ('confirmed', 'lost', 'removed')",
            name="ck_video_tracklet_state",
        ),
        CheckConstraint("first_frame_index <= last_frame_index", name="ck_video_tracklet_frame_order"),
        CheckConstraint("first_pts_ns <= last_pts_ns", name="ck_video_tracklet_pts_order"),
        CheckConstraint("observation_count >= 0", name="ck_video_tracklet_observation_count_nonnegative"),
        CheckConstraint(
            "valid_embedding_count >= 0 AND valid_embedding_count <= observation_count",
            name="ck_video_tracklet_embedding_count_consistent",
        ),
    )

    tracklet_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_job.job_id", ondelete="RESTRICT"), nullable=False
    )
    track_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_track.track_id", ondelete="RESTRICT"), nullable=False
    )
    tracklet_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    first_frame_index: Mapped[int] = mapped_column(BigInteger, nullable=False)
    last_frame_index: Mapped[int] = mapped_column(BigInteger, nullable=False)
    first_pts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    last_pts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    valid_embedding_count: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    mean_quality: Mapped[float | None] = mapped_column(REAL, nullable=True)
    max_quality: Mapped[float | None] = mapped_column(REAL, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
