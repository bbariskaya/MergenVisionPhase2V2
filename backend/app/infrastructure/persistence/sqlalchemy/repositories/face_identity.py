"""SQLAlchemy face identity repository."""

from __future__ import annotations

from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities.face_identity import FaceIdentity
from app.domain.errors import ConcurrentUpdateError
from app.domain.value_objects import FaceId
from app.infrastructure.persistence.sqlalchemy.models.face_identity import FaceIdentityOrm


def _to_domain(orm: FaceIdentityOrm) -> FaceIdentity:
    return FaceIdentity(
        face_id=FaceId(orm.face_id),
        status=orm.status,
        is_active=orm.is_active,
        display_name=orm.display_name,
        identity_metadata=dict(orm.identity_metadata or {}),
        version=orm.version,
        created_at=orm.created_at,
        updated_at=orm.updated_at,
        deleted_at=orm.deleted_at,
    )


class SqlAlchemyFaceIdentityRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, identity: FaceIdentity) -> None:
        orm = FaceIdentityOrm(
            face_id=identity.face_id,
            status=identity.status,
            is_active=identity.is_active,
            display_name=identity.display_name,
            identity_metadata=identity.identity_metadata,
            version=identity.version,
            created_at=identity.created_at,
            updated_at=identity.updated_at,
            deleted_at=identity.deleted_at,
        )
        self._session.add(orm)

    async def get_by_id(self, face_id: FaceId) -> FaceIdentity | None:
        result = await self._session.execute(
            select(FaceIdentityOrm).where(FaceIdentityOrm.face_id == face_id)
        )
        orm = result.scalar_one_or_none()
        return _to_domain(orm) if orm else None

    async def get_active_by_id(self, face_id: FaceId) -> FaceIdentity | None:
        result = await self._session.execute(
            select(FaceIdentityOrm)
            .where(FaceIdentityOrm.face_id == face_id)
            .where(FaceIdentityOrm.is_active.is_(True))
        )
        orm = result.scalar_one_or_none()
        return _to_domain(orm) if orm else None

    async def update(self, identity: FaceIdentity) -> None:
        result = await self._session.execute(
            select(FaceIdentityOrm).where(FaceIdentityOrm.face_id == identity.face_id)
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            raise ValueError(f"FaceIdentity {identity.face_id} not found")
        orm.status = identity.status
        orm.is_active = identity.is_active
        orm.display_name = identity.display_name
        orm.identity_metadata = identity.identity_metadata
        orm.version = identity.version
        orm.updated_at = identity.updated_at
        orm.deleted_at = identity.deleted_at

    async def update_with_expected_version(
        self,
        identity: FaceIdentity,
        expected_version: int,
    ) -> FaceIdentity:
        result = await self._session.execute(
            update(FaceIdentityOrm)
            .where(
                FaceIdentityOrm.face_id == identity.face_id,
                FaceIdentityOrm.version == expected_version,
            )
            .values(
                status=identity.status,
                is_active=identity.is_active,
                display_name=identity.display_name,
                identity_metadata=identity.identity_metadata,
                version=FaceIdentityOrm.version + 1,
                updated_at=identity.updated_at,
                deleted_at=identity.deleted_at,
            )
            .returning(FaceIdentityOrm.version)
        )
        new_version = result.scalar_one_or_none()
        if new_version is None:
            raise ConcurrentUpdateError(
                f"FaceIdentity {identity.face_id} was modified concurrently"
            )
        return FaceIdentity(
            face_id=FaceId(identity.face_id),
            status=identity.status,
            is_active=identity.is_active,
            display_name=identity.display_name,
            identity_metadata=dict(identity.identity_metadata),
            version=new_version,
            created_at=identity.created_at,
            updated_at=identity.updated_at,
            deleted_at=identity.deleted_at,
        )

    async def list_all(self) -> list[FaceIdentity]:
        result = await self._session.execute(
            select(FaceIdentityOrm).order_by(desc(FaceIdentityOrm.created_at))
        )
        return [_to_domain(orm) for orm in result.scalars().all()]
