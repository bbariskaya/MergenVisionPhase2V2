"""SQLAlchemy mapping for recognition_result table."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Numeric, String, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.sqlalchemy.base import Base


class RecognitionResultOrm(Base):
    __tablename__ = "recognition_result"

    __table_args__ = (
        CheckConstraint(
            "status IN ('known', 'anonymous', 'new_anonymous')",
            name="ck_recognition_result_status",
        ),
    )

    result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    process_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("process_record.process_id", ondelete="RESTRICT"),
        nullable=False,
    )
    face_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("face_identity.face_id", ondelete="RESTRICT"),
        nullable=False,
    )
    sample_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("face_sample.sample_id", ondelete="RESTRICT"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    bounding_box: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    match_confidence: Mapped[float] = mapped_column(Numeric(precision=4, scale=3), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    result_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
