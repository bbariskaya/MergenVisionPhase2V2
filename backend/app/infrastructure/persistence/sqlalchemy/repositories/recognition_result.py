"""SQLAlchemy recognition result repository."""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities.recognition_result import RecognitionResult
from app.domain.value_objects import BoundingBox, FaceId, ProcessId, ResultId, SampleId
from app.infrastructure.persistence.sqlalchemy.models.recognition_result import (
    RecognitionResultOrm,
)


def _to_domain(orm: RecognitionResultOrm) -> RecognitionResult:
    return RecognitionResult(
        result_id=ResultId(orm.result_id),
        process_id=ProcessId(orm.process_id),
        face_id=FaceId(orm.face_id),
        sample_id=SampleId(orm.sample_id) if orm.sample_id else None,
        status=orm.status,
        bounding_box=BoundingBox(**orm.bounding_box),
        match_confidence=float(orm.match_confidence),
        created_at=orm.created_at,
        metadata=dict(orm.result_metadata or {}),
    )


class SqlAlchemyRecognitionResultRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, result: RecognitionResult) -> None:
        orm = RecognitionResultOrm(
            result_id=result.result_id,
            process_id=result.process_id,
            face_id=result.face_id,
            sample_id=result.sample_id,
            status=result.status,
            bounding_box={
                "x": result.bounding_box.x,
                "y": result.bounding_box.y,
                "width": result.bounding_box.width,
                "height": result.bounding_box.height,
            },
            match_confidence=Decimal(str(result.match_confidence)),
            created_at=result.created_at,
            result_metadata=result.metadata,
        )
        self._session.add(orm)

    async def list_by_process_id(self, process_id: ProcessId) -> list[RecognitionResult]:
        result = await self._session.execute(
            select(RecognitionResultOrm)
            .where(RecognitionResultOrm.process_id == process_id)
            .order_by(RecognitionResultOrm.created_at)
        )
        return [_to_domain(orm) for orm in result.scalars().all()]
