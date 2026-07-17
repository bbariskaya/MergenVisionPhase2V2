"""SQLAlchemy mapping for video_asset table."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, CheckConstraint, DateTime, Integer, String, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.sqlalchemy.base import Base


class VideoAssetOrm(Base):
    __tablename__ = "video_asset"

    __table_args__ = (
        CheckConstraint(
            "state IN ('uploading', 'validating', 'ready', 'rejected', 'deleting', 'deleted')",
            name="ck_video_asset_state",
        ),
        CheckConstraint("size_bytes IS NULL OR size_bytes >= 0", name="ck_video_asset_size_nonnegative"),
        CheckConstraint("display_width IS NULL OR display_width > 0", name="ck_video_asset_width_positive"),
        CheckConstraint("display_height IS NULL OR display_height > 0", name="ck_video_asset_height_positive"),
        CheckConstraint("duration_ns IS NULL OR duration_ns >= 0", name="ck_video_asset_duration_nonnegative"),
        CheckConstraint("time_base_den IS NULL OR time_base_den > 0", name="ck_video_asset_time_base_den_positive"),
        CheckConstraint(
            "nominal_fps_den IS NULL OR nominal_fps_den > 0", name="ck_video_asset_fps_den_positive"
        ),
        CheckConstraint(
            """
            state != 'ready' OR (
                bucket IS NOT NULL AND btrim(bucket) != ''
                AND object_key IS NOT NULL AND btrim(object_key) != ''
                AND content_sha256 IS NOT NULL AND btrim(content_sha256) != ''
                AND size_bytes IS NOT NULL
                AND container_format IS NOT NULL AND btrim(container_format) != ''
                AND video_codec IS NOT NULL AND btrim(video_codec) != ''
                AND display_width IS NOT NULL AND display_width > 0
                AND display_height IS NOT NULL AND display_height > 0
                AND duration_ns IS NOT NULL AND duration_ns >= 0
                AND time_base_num IS NOT NULL AND time_base_den IS NOT NULL
                AND retention_until IS NOT NULL
            )
            """,
            name="ck_video_asset_ready_requires_fields",
        ),
    )

    video_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    upload_session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, unique=True
    )
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    staging_bucket: Mapped[str | None] = mapped_column(String(255), nullable=True)
    staging_object_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    bucket: Mapped[str | None] = mapped_column(String(255), nullable=True)
    object_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    content_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    container_format: Mapped[str | None] = mapped_column(String(64), nullable=True)
    video_codec: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pixel_format: Mapped[str | None] = mapped_column(String(64), nullable=True)
    display_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    display_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rotation_degrees: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    duration_ns: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    time_base_num: Mapped[int | None] = mapped_column(Integer, nullable=True)
    time_base_den: Mapped[int | None] = mapped_column(Integer, nullable=True)
    nominal_fps_num: Mapped[int | None] = mapped_column(Integer, nullable=True)
    nominal_fps_den: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_frames: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    retention_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    ready_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
