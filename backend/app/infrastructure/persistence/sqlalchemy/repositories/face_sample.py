"""SQLAlchemy face sample repository."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities.face_sample import FaceSample
from app.domain.value_objects import FaceId, SampleId
from app.infrastructure.persistence.sqlalchemy.models.face_sample import FaceSampleOrm


def _to_domain(orm: FaceSampleOrm) -> FaceSample:
    return FaceSample(
        sample_id=SampleId(orm.sample_id),
        face_id=FaceId(orm.face_id),
        state=orm.state,
        bucket=orm.bucket,
        object_key=orm.object_key,
        failure_code=orm.failure_code,
        is_active=orm.is_active,
        created_at=orm.created_at,
        activated_at=orm.activated_at,
        deactivated_at=orm.deactivated_at,
    )


class SqlAlchemyFaceSampleRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, sample: FaceSample) -> None:
        orm = FaceSampleOrm(
            sample_id=sample.sample_id,
            face_id=sample.face_id,
            state=sample.state,
            bucket=sample.bucket,
            object_key=sample.object_key,
            failure_code=sample.failure_code,
            is_active=sample.is_active,
            created_at=sample.created_at,
            activated_at=sample.activated_at,
            deactivated_at=sample.deactivated_at,
        )
        self._session.add(orm)

    async def get_by_id(self, sample_id: SampleId) -> FaceSample | None:
        result = await self._session.execute(
            select(FaceSampleOrm).where(FaceSampleOrm.sample_id == sample_id)
        )
        orm = result.scalar_one_or_none()
        return _to_domain(orm) if orm else None

    async def list_active_by_face_id(self, face_id: FaceId) -> list[FaceSample]:
        result = await self._session.execute(
            select(FaceSampleOrm)
            .where(FaceSampleOrm.face_id == face_id)
            .where(FaceSampleOrm.is_active.is_(True))
            .order_by(FaceSampleOrm.created_at)
        )
        return [_to_domain(orm) for orm in result.scalars().all()]

    async def update(self, sample: FaceSample) -> None:
        result = await self._session.execute(
            select(FaceSampleOrm).where(FaceSampleOrm.sample_id == sample.sample_id)
        )
        orm = result.scalar_one_or_none()
        if orm is None:
            raise ValueError(f"FaceSample {sample.sample_id} not found")
        orm.state = sample.state
        orm.bucket = sample.bucket
        orm.object_key = sample.object_key
        orm.failure_code = sample.failure_code
        orm.is_active = sample.is_active
        orm.activated_at = sample.activated_at
        orm.deactivated_at = sample.deactivated_at
