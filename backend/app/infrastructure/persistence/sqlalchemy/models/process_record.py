"""SQLAlchemy mapping for process_record table."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Integer, String, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.sqlalchemy.base import Base


class ProcessRecordOrm(Base):
    __tablename__ = "process_record"

    __table_args__ = (
        CheckConstraint(
            "process_type IN ('image_recognize', 'face_enroll', 'face_delete')",
            name="ck_process_record_type",
        ),
        CheckConstraint(
            "status IN ('processing', 'completed', 'failed')",
            name="ck_process_record_status",
        ),
        CheckConstraint(
            """
            (status = 'processing' AND completed_at IS NULL AND face_count IS NULL AND error_code IS NULL)
            OR
            (status = 'completed' AND completed_at IS NOT NULL AND face_count IS NOT NULL
             AND face_count >= 0 AND error_code IS NULL)
            OR
            (status = 'failed' AND completed_at IS NOT NULL AND error_code IS NOT NULL
             AND btrim(error_code) != '')
            """,
            name="ck_process_record_lifecycle",
        ),
    )

    process_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    process_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    face_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
