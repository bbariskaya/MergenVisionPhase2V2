"""SQLAlchemy mapping for appearance_interval table."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.sqlalchemy.base import Base


class AppearanceIntervalOrm(Base):
    __tablename__ = "appearance_interval"

    __table_args__ = (
        UniqueConstraint("track_id", "interval_index", name="uq_appearance_track_interval"),
        CheckConstraint("start_frame_index <= end_frame_index", name="ck_appearance_frame_order"),
        CheckConstraint("start_pts_ns <= end_pts_ns", name="ck_appearance_pts_order"),
        CheckConstraint("detection_count >= 0", name="ck_appearance_detection_count_nonnegative"),
    )

    appearance_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_job.job_id", ondelete="RESTRICT"), nullable=False
    )
    track_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("video_track.track_id", ondelete="RESTRICT"), nullable=False
    )
    interval_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_frame_index: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_frame_index: Mapped[int] = mapped_column(BigInteger, nullable=False)
    start_pts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_pts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    detection_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
