"""PostgreSQL persistence store for Phase 2's existing tables.

This store connects to the running Phase 2 PostgreSQL and writes directly into
``person``, ``face_identity`` and ``face_sample`` using Phase 2's own ORM models
when available. No new tables or migrations are created by Phase 1.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import UUID, insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from mv_phase1_bulk.types import FaceRecord, PersonRecord, SampleRecord

# Phase 2 models are reused read-only so the schema can never drift.
# If the backend package is importable (e.g. when running inside the Phase 2
# repository or Docker context), use its declarative models. Otherwise fall back
# to an identical SQLAlchemy Core mirror so the package stays self-contained.
_PHASE2_BACKEND = Path("/home/user/Workspace/MergenVisionPhase2v2/backend")
if str(_PHASE2_BACKEND) not in sys.path:
    sys.path.insert(0, str(_PHASE2_BACKEND))

try:
    from app.infrastructure.persistence.sqlalchemy.models.person import PersonOrm  # type: ignore[import]
    from app.infrastructure.persistence.sqlalchemy.models.face_identity import FaceIdentityOrm  # type: ignore[import]
    from app.infrastructure.persistence.sqlalchemy.models.face_sample import FaceSampleOrm  # type: ignore[import]

    _USE_PHASE2_MODELS = True
    person_table = PersonOrm.__table__
    face_identity_table = FaceIdentityOrm.__table__
    face_sample_table = FaceSampleOrm.__table__
    metadata = PersonOrm.metadata
except Exception:  # pragma: no cover - fallback only
    _USE_PHASE2_MODELS = False
    metadata = MetaData()

    person_table = Table(
        "person",
        metadata,
        Column("person_id", UUID(as_uuid=True), primary_key=True),
        Column("display_name", String(255), nullable=False),
        Column("status", String(32), nullable=False, default="active"),
        Column("metadata", JSON, nullable=False, default=dict),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
        Column("deleted_at", DateTime(timezone=True), nullable=True),
    )

    face_identity_table = Table(
        "face_identity",
        metadata,
        Column("face_id", UUID(as_uuid=True), primary_key=True),
        Column("person_id", UUID(as_uuid=True), ForeignKey("person.person_id"), nullable=False),
        Column("model_version", String(64), nullable=False),
        Column("status", String(32), nullable=False, default="active"),
        Column("is_canonical", Boolean, nullable=False, default=True),
        Column("display_name", String(255), nullable=False),
        Column("metadata", JSON, nullable=False, default=dict),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
        Column("deleted_at", DateTime(timezone=True), nullable=True),
        UniqueConstraint("person_id", "model_version", name="uq_face_identity_person_model"),
    )

    face_sample_table = Table(
        "face_sample",
        metadata,
        Column("sample_id", UUID(as_uuid=True), primary_key=True),
        Column("face_id", UUID(as_uuid=True), ForeignKey("face_identity.face_id"), nullable=False),
        Column("person_id", UUID(as_uuid=True), ForeignKey("person.person_id"), nullable=False),
        Column("status", String(32), nullable=False, default="pending"),
        Column("bucket", String(128), nullable=True),
        Column("object_key", String(512), nullable=True),
        Column("sha256", String(64), nullable=False),
        Column("model_version", String(64), nullable=False),
        Column("preprocess_version", String(32), nullable=False),
        Column("rejection_reason", String(128), nullable=True),
        Column("metadata", JSON, nullable=False, default=dict),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
        Column("deleted_at", DateTime(timezone=True), nullable=True),
        UniqueConstraint("face_id", "sha256", name="uq_face_sample_face_sha256"),
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PostgresStore:
    """Async PostgreSQL store that writes into Phase 2's existing tables."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._engine: AsyncEngine | None = None

    async def connect(self) -> None:
        if self._engine is None:
            self._engine = create_async_engine(
                self._database_url,
                future=True,
                echo=False,
                pool_pre_ping=True,
            )

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("PostgresStore not connected")
        return self._engine

    # ------------------------------------------------------------------
    # Bulk upsert helpers
    # ------------------------------------------------------------------
    async def upsert_people(
        self,
        conn: AsyncConnection,
        people: Sequence[PersonRecord],
    ) -> None:
        if not people:
            return
        now = _utcnow()
        rows = [
            {
                "person_id": p.person_id,
                "display_name": p.display_name,
                "status": p.status,
                "metadata": p.metadata,
                "created_at": now,
                "updated_at": now,
                "deleted_at": None,
            }
            for p in people
        ]
        stmt = pg_insert(person_table).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["person_id"],
            set_={
                "display_name": stmt.excluded.display_name,
                "status": stmt.excluded.status,
                "metadata": stmt.excluded.metadata,
                "updated_at": stmt.excluded.updated_at,
                "deleted_at": None,
            },
        )
        await conn.execute(stmt)

    async def upsert_faces(
        self,
        conn: AsyncConnection,
        faces: Sequence[FaceRecord],
    ) -> None:
        if not faces:
            return
        now = _utcnow()
        rows = [
            {
                "face_id": f.face_id,
                "person_id": f.person_id,
                "model_version": f.model_version,
                "status": f.status,
                "is_canonical": f.is_canonical,
                "display_name": f.display_name,
                "metadata": f.metadata,
                "created_at": now,
                "updated_at": now,
                "deleted_at": None,
            }
            for f in faces
        ]
        stmt = pg_insert(face_identity_table).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["face_id"],
            set_={
                "status": stmt.excluded.status,
                "is_canonical": stmt.excluded.is_canonical,
                "display_name": stmt.excluded.display_name,
                "metadata": stmt.excluded.metadata,
                "updated_at": stmt.excluded.updated_at,
                "deleted_at": None,
            },
        )
        await conn.execute(stmt)

    async def upsert_samples_pending(
        self,
        conn: AsyncConnection,
        samples: Sequence[SampleRecord],
    ) -> None:
        """Insert or re-touch samples in ``pending`` state for idempotent retries."""
        if not samples:
            return
        now = _utcnow()
        rows = [
            {
                "sample_id": s.sample_id,
                "face_id": s.face_id,
                "person_id": s.person_id,
                "status": "pending",
                "bucket": s.bucket,
                "object_key": s.object_key,
                "sha256": s.sha256,
                "model_version": s.model_version,
                "preprocess_version": s.preprocess_version,
                "rejection_reason": s.rejection_reason,
                "metadata": s.metadata,
                "created_at": now,
                "updated_at": now,
                "deleted_at": None,
            }
            for s in samples
        ]
        stmt = pg_insert(face_sample_table).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["sample_id"],
            set_={
                "status": "pending",
                "bucket": stmt.excluded.bucket,
                "object_key": stmt.excluded.object_key,
                "sha256": stmt.excluded.sha256,
                "model_version": stmt.excluded.model_version,
                "preprocess_version": stmt.excluded.preprocess_version,
                "rejection_reason": stmt.excluded.rejection_reason,
                "metadata": stmt.excluded.metadata,
                "updated_at": stmt.excluded.updated_at,
                "deleted_at": None,
            },
        )
        await conn.execute(stmt)

    async def activate_samples(
        self,
        conn: AsyncConnection,
        sample_ids: Sequence[str],
    ) -> None:
        if not sample_ids:
            return
        await conn.execute(
            update(face_sample_table)
            .where(face_sample_table.c.sample_id.in_(sample_ids))
            .values(status="active", updated_at=_utcnow(), rejection_reason=None),
        )

    async def fail_samples(
        self,
        conn: AsyncConnection,
        sample_ids: Sequence[str],
        reason: str,
    ) -> None:
        if not sample_ids:
            return
        await conn.execute(
            update(face_sample_table)
            .where(face_sample_table.c.sample_id.in_(sample_ids))
            .values(status="failed", updated_at=_utcnow(), rejection_reason=reason),
        )

    async def get_samples_by_status(
        self,
        conn: AsyncConnection,
        status: str,
        limit: int = 10000,
    ) -> list[SampleRecord]:
        result = await conn.execute(
            select(
                face_sample_table.c.sample_id,
                face_sample_table.c.face_id,
                face_sample_table.c.person_id,
                face_sample_table.c.status,
                face_sample_table.c.bucket,
                face_sample_table.c.object_key,
                face_sample_table.c.sha256,
                face_sample_table.c.model_version,
                face_sample_table.c.preprocess_version,
                face_sample_table.c.rejection_reason,
                face_sample_table.c.metadata,
            )
            .where(face_sample_table.c.status == status)
            .limit(limit)
        )
        return [
            SampleRecord(
                sample_id=str(row.sample_id),
                face_id=str(row.face_id),
                person_id=str(row.person_id),
                status=row.status,
                bucket=row.bucket,
                object_key=row.object_key,
                sha256=row.sha256,
                model_version=row.model_version,
                preprocess_version=row.preprocess_version,
                rejection_reason=row.rejection_reason,
                metadata=dict(row.metadata or {}),
            )
            for row in result.mappings()
        ]

    # ------------------------------------------------------------------
    # Transaction-scoped public API
    # ------------------------------------------------------------------
    async def prepare_enrollment(
        self,
        people: Sequence[PersonRecord],
        faces: Sequence[FaceRecord],
        samples: Sequence[SampleRecord],
    ) -> None:
        """Write pending person/face/sample rows inside one transaction."""
        async with self._engine.begin() as conn:  # type: ignore[union-attr]
            await self.upsert_people(conn, people)
            await self.upsert_faces(conn, faces)
            await self.upsert_samples_pending(conn, samples)

    async def get_pending_samples(self, limit: int = 10000) -> list[SampleRecord]:
        async with self._engine.connect() as conn:  # type: ignore[union-attr]
            return await self.get_samples_by_status(conn, "pending", limit=limit)
