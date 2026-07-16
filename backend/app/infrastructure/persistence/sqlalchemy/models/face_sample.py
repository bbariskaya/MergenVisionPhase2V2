"""SQLAlchemy mapping for face_sample table."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.sqlalchemy.base import Base


class FaceSampleOrm(Base):
    __tablename__ = "face_sample"

    __table_args__ = (
        CheckConstraint(
            "state IN ('pending', 'active', 'failed', 'inactive')",
            name="ck_face_sample_state",
        ),
        CheckConstraint(
            """
            (state = 'pending' AND is_active = false AND bucket IS NULL AND object_key IS NULL
             AND activated_at IS NULL AND failure_code IS NULL AND deactivated_at IS NULL)
            OR
            (state = 'active' AND is_active = true AND bucket IS NOT NULL
             AND btrim(bucket) != '' AND object_key IS NOT NULL AND btrim(object_key) != ''
             AND activated_at IS NOT NULL AND failure_code IS NULL AND deactivated_at IS NULL)
            OR
            (state = 'failed' AND is_active = false AND failure_code IS NOT NULL
             AND btrim(failure_code) != '')
            OR
            (state = 'inactive' AND is_active = false AND bucket IS NOT NULL
             AND btrim(bucket) != '' AND object_key IS NOT NULL AND btrim(object_key) != ''
             AND activated_at IS NOT NULL AND deactivated_at IS NOT NULL)
            """,
            name="ck_face_sample_lifecycle",
        ),
    )

    sample_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    face_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("face_identity.face_id", ondelete="RESTRICT"),
        nullable=False,
    )
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    bucket: Mapped[str | None] = mapped_column(String(255), nullable=True)
    object_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deactivated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
