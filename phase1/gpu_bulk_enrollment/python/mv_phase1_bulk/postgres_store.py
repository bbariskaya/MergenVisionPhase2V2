"""PostgreSQL persistence store for Phase 2's existing tables.

This store connects to the running Phase 2 PostgreSQL and writes directly into
``face_identity`` and ``face_sample`` using SQLAlchemy Core. No new tables or
migrations are created by Phase 1, and Person/redirect semantics are intentionally
absent: the single source of truth for identity is ``face_identity``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    func,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from mv_phase1_bulk.types import FaceRecord, SampleRecord

metadata = MetaData()

face_identity_table = Table(
    "face_identity",
    metadata,
    Column("face_id", UUID(as_uuid=True), primary_key=True),
    Column("status", String(16), nullable=False),
    Column("is_active", Boolean, nullable=False, default=True),
    Column("display_name", String(255), nullable=True),
    Column("identity_metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("version", Integer, nullable=False, default=1),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()),
    Column("deleted_at", DateTime(timezone=True), nullable=True),
)

face_sample_table = Table(
    "face_sample",
    metadata,
    Column("sample_id", UUID(as_uuid=True), primary_key=True),
    Column("face_id", UUID(as_uuid=True), ForeignKey("face_identity.face_id", ondelete="RESTRICT"), nullable=False),
    Column("state", String(16), nullable=False),
    Column("bucket", String(255), nullable=True),
    Column("object_key", String(1024), nullable=True),
    Column("failure_code", String(64), nullable=True),
    Column("is_active", Boolean, nullable=False, default=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("activated_at", DateTime(timezone=True), nullable=True),
    Column("deactivated_at", DateTime(timezone=True), nullable=True),
    UniqueConstraint("face_id", "object_key", name="uq_face_sample_face_object_key"),
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


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
    # Conflict-safe reads
    # ------------------------------------------------------------------
    async def get_face_identity(self, conn: AsyncConnection, face_id: str) -> FaceRecord | None:
        row = await conn.execute(
            select(
                face_identity_table.c.face_id,
                face_identity_table.c.status,
                face_identity_table.c.is_active,
                face_identity_table.c.display_name,
                face_identity_table.c.identity_metadata,
                face_identity_table.c.version,
            ).where(face_identity_table.c.face_id == face_id)
        )
        r = row.mappings().first()
        if r is None:
            return None
        return FaceRecord(
            face_id=str(r.face_id),
            status=r.status,
            is_active=r.is_active,
            display_name=r.display_name or "",
            identity_metadata=dict(r.identity_metadata or {}),
            version=r.version,
        )

    async def get_sample(self, conn: AsyncConnection, sample_id: str) -> SampleRecord | None:
        row = await conn.execute(
            select(
                face_sample_table.c.sample_id,
                face_sample_table.c.face_id,
                face_sample_table.c.state,
                face_sample_table.c.bucket,
                face_sample_table.c.object_key,
                face_sample_table.c.failure_code,
                face_sample_table.c.is_active,
                face_sample_table.c.activated_at,
                face_sample_table.c.deactivated_at,
            ).where(face_sample_table.c.sample_id == sample_id)
        )
        r = row.mappings().first()
        if r is None:
            return None
        return SampleRecord(
            sample_id=str(r.sample_id),
            face_id=str(r.face_id),
            state=r.state,
            bucket=r.bucket,
            object_key=r.object_key,
            failure_code=r.failure_code,
            is_active=r.is_active,
            activated_at=r.activated_at.isoformat() if r.activated_at else None,
            deactivated_at=r.deactivated_at.isoformat() if r.deactivated_at else None,
        )

    # ------------------------------------------------------------------
    # Bulk upsert helpers
    # ------------------------------------------------------------------
    async def upsert_faces(
        self,
        conn: AsyncConnection,
        faces: Sequence[FaceRecord],
    ) -> None:
        if not faces:
            return
        rows = [
            {
                "face_id": f.face_id,
                "status": f.status,
                "is_active": f.is_active,
                "display_name": f.display_name,
                "identity_metadata": f.identity_metadata,
                "version": f.version,
            }
            for f in faces
        ]
        stmt = pg_insert(face_identity_table).values(rows)
        # On conflict, reactivate a previously soft-deleted identity without
        # overwriting name/metadata from a prior successful enrollment.
        stmt = stmt.on_conflict_do_update(
            index_elements=["face_id"],
            set_={
                "is_active": True,
                "deleted_at": None,
                "updated_at": _utcnow(),
            },
        )
        await conn.execute(stmt)

    async def upsert_samples_pending(
        self,
        conn: AsyncConnection,
        samples: Sequence[SampleRecord],
    ) -> None:
        """Insert pending samples only if they do not already exist.

        Existing active/inactive/failed samples are never downgraded to pending.
        """
        if not samples:
            return
        rows = [
            {
                "sample_id": s.sample_id,
                "face_id": s.face_id,
                "state": "pending",
                "bucket": None,
                "object_key": None,
                "failure_code": None,
                "is_active": False,
            }
            for s in samples
        ]
        stmt = pg_insert(face_sample_table).values(rows)
        stmt = stmt.on_conflict_do_nothing(index_elements=["sample_id"])
        await conn.execute(stmt)

    async def activate_samples(
        self,
        conn: AsyncConnection,
        activations: Sequence[tuple[str, str, str]],
    ) -> None:
        """Activate samples with their own object keys.

        Each tuple is ``(sample_id, bucket, object_key)``.  The update is done
        per-row because object keys differ per sample.
        """
        if not activations:
            return
        activated_at = _utcnow()
        for sample_id, bucket, object_key in activations:
            await conn.execute(
                update(face_sample_table)
                .where(face_sample_table.c.sample_id == sample_id)
                .values(
                    state="active",
                    is_active=True,
                    bucket=bucket,
                    object_key=object_key,
                    activated_at=activated_at,
                    failure_code=None,
                ),
            )

    async def fail_samples(
        self,
        conn: AsyncConnection,
        sample_ids: Sequence[str],
        failure_code: str,
    ) -> None:
        if not sample_ids:
            return
        await conn.execute(
            update(face_sample_table)
            .where(face_sample_table.c.sample_id.in_(sample_ids))
            .values(
                state="failed",
                is_active=False,
                failure_code=failure_code,
            ),
        )

    async def get_samples_by_state(
        self,
        conn: AsyncConnection,
        state: str,
        limit: int = 10000,
    ) -> list[SampleRecord]:
        result = await conn.execute(
            select(
                face_sample_table.c.sample_id,
                face_sample_table.c.face_id,
                face_sample_table.c.state,
                face_sample_table.c.bucket,
                face_sample_table.c.object_key,
                face_sample_table.c.failure_code,
                face_sample_table.c.is_active,
                face_sample_table.c.activated_at,
                face_sample_table.c.deactivated_at,
            )
            .where(face_sample_table.c.state == state)
            .limit(limit)
        )
        return [
            SampleRecord(
                sample_id=str(row.sample_id),
                face_id=str(row.face_id),
                state=row.state,
                bucket=row.bucket,
                object_key=row.object_key,
                failure_code=row.failure_code,
                is_active=row.is_active,
                activated_at=row.activated_at.isoformat() if row.activated_at else None,
                deactivated_at=row.deactivated_at.isoformat() if row.deactivated_at else None,
            )
            for row in result.mappings()
        ]

    # ------------------------------------------------------------------
    # Transaction-scoped public API
    # ------------------------------------------------------------------
    async def prepare_enrollment(
        self,
        faces: Sequence[FaceRecord],
        samples: Sequence[SampleRecord],
    ) -> None:
        """Write pending face/sample rows inside one transaction.

        Existing rows are left untouched; create-if-absent semantics keep bulk
        enrollment idempotent.
        """
        async with self._engine.begin() as conn:  # type: ignore[union-attr]
            await self.upsert_faces(conn, faces)
            await self.upsert_samples_pending(conn, samples)

    async def get_pending_samples(self, limit: int = 10000) -> list[SampleRecord]:
        async with self._engine.connect() as conn:  # type: ignore[union-attr]
            return await self.get_samples_by_state(conn, "pending", limit=limit)

    async def activate_samples_tx(
        self,
        activations: Sequence[tuple[str, str, str]],
    ) -> None:
        """Activate samples inside a fresh transaction."""
        async with self._engine.begin() as conn:  # type: ignore[union-attr]
            await self.activate_samples(conn, activations)

    async def fail_samples_tx(
        self,
        failures: Sequence[tuple[str, str]],
    ) -> None:
        """Mark samples as failed inside a fresh transaction."""
        async with self._engine.begin() as conn:  # type: ignore[union-attr]
            for sample_id, failure_code in failures:
                await self.fail_samples(conn, [sample_id], failure_code)

    async def deactivate_face_if_no_active_samples_tx(self, face_id: str) -> None:
        """Deactivate a face identity only if it has no active samples.

        Used as a fail-safe when every sample for a subject failed to persist.
        If the identity already has active samples from a previous run, it is
        left untouched.
        """
        async with self._engine.begin() as conn:  # type: ignore[union-attr]
            active_count = await conn.scalar(
                select(func.count()).where(
                    face_sample_table.c.face_id == face_id,
                    face_sample_table.c.state == "active",
                )
            )
            if active_count == 0:
                await conn.execute(
                    update(face_identity_table)
                    .where(face_identity_table.c.face_id == face_id)
                    .values(is_active=False, deleted_at=_utcnow(), updated_at=_utcnow())
                )

    async def get_samples_for_face_ids(
        self,
        face_ids: Sequence[str],
    ) -> list[SampleRecord]:
        """Return every face_sample row for the given face ids."""
        if not face_ids:
            return []
        async with self._engine.connect() as conn:  # type: ignore[union-attr]
            result = await conn.execute(
                select(
                    face_sample_table.c.sample_id,
                    face_sample_table.c.face_id,
                    face_sample_table.c.state,
                    face_sample_table.c.bucket,
                    face_sample_table.c.object_key,
                    face_sample_table.c.failure_code,
                    face_sample_table.c.is_active,
                    face_sample_table.c.activated_at,
                    face_sample_table.c.deactivated_at,
                ).where(face_sample_table.c.face_id.in_(face_ids))
            )
            return [
                SampleRecord(
                    sample_id=str(row.sample_id),
                    face_id=str(row.face_id),
                    state=row.state,
                    bucket=row.bucket,
                    object_key=row.object_key,
                    failure_code=row.failure_code,
                    is_active=row.is_active,
                    activated_at=row.activated_at.isoformat() if row.activated_at else None,
                    deactivated_at=row.deactivated_at.isoformat() if row.deactivated_at else None,
                )
                for row in result.mappings()
            ]
