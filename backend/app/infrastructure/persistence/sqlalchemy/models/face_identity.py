"""SQLAlchemy mapping for face_identity table."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, CheckConstraint, DateTime, Integer, String, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.sqlalchemy.base import Base


class FaceIdentityOrm(Base):
    __tablename__ = "face_identity"

    __table_args__ = (
        CheckConstraint("status IN ('anonymous', 'known')", name="ck_face_identity_status"),
        CheckConstraint("version >= 1", name="ck_face_identity_version_positive"),
        CheckConstraint(
            "status != 'known' OR (display_name IS NOT NULL AND btrim(display_name) != '')",
            name="ck_face_identity_known_name",
        ),
        CheckConstraint(
            "(is_active = true AND deleted_at IS NULL) OR (is_active = false AND deleted_at IS NOT NULL)",
            name="ck_face_identity_active_deleted",
        ),
    )

    face_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    identity_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
