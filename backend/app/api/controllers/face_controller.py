"""HTTP-facing controllers for the face recognition API.

Controllers map between HTTP request/response DTOs and the application service
layer. They contain no business rules beyond trivial validation/formatting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from app.api.schemas import (
    BoundingBoxSchema,
    EnrollRequest,
    EnrollResponse,
    FaceHistoryResponse,
    FaceResponse,
    HistoryEntry,
    IdentityDetailResponse,
    ProcessResponse,
    RecognizeResponse,
)
from app.application.services.image_recognition_service import (
    ImageRecognitionService,
    RecognitionResultItem,
)
from app.domain.entities.process_record import ProcessRecord
from app.domain.errors import ValidationError
from app.domain.value_objects import FaceId, ProcessId


@dataclass(frozen=True)
class RecognizeRequestData:
    image_bytes: bytes
    filename: str | None


class FaceController:
    def __init__(self, service: ImageRecognitionService) -> None:
        self._service = service

    async def recognize(self, data: RecognizeRequestData) -> RecognizeResponse:
        result = await self._service.recognize_image(data.image_bytes)
        process = result.process
        return RecognizeResponse(
            process_id=str(process.process_id),
            status=process.status,
            face_count=len(result.faces),
            faces=[self._to_face_response(f) for f in result.faces],
        )

    async def enroll(self, request: EnrollRequest) -> EnrollResponse:
        try:
            face_id = FaceId(UUID(request.face_id))
        except ValueError as exc:
            raise ValidationError("face_id must be a valid UUID") from exc

        identity = await self._service.enroll_face(
            face_id=face_id,
            display_name=request.name,
            metadata=request.metadata or {},
        )
        return EnrollResponse(
            face_id=str(identity.face_id),
            status=identity.status,
            name=identity.display_name or request.name,
            metadata=dict(identity.identity_metadata),
        )

    async def get_identity(self, face_id_str: str) -> IdentityDetailResponse | None:
        try:
            face_id = FaceId(UUID(face_id_str))
        except ValueError as exc:
            raise ValidationError("face_id must be a valid UUID") from exc

        identity = await self._service.get_identity_detail(face_id)
        if identity is None:
            return None
        return IdentityDetailResponse(
            face_id=str(identity.face_id),
            status=identity.status,
            name=identity.display_name,
            metadata=dict(identity.identity_metadata),
            created_at=self._format_dt(identity.created_at),
            updated_at=self._format_dt(identity.updated_at),
        )

    async def delete_identity(self, face_id_str: str) -> bool:
        try:
            face_id = FaceId(UUID(face_id_str))
        except ValueError as exc:
            raise ValidationError("face_id must be a valid UUID") from exc

        await self._service.delete_identity(face_id)
        return True

    async def get_history(self, face_id_str: str) -> FaceHistoryResponse:
        try:
            face_id = FaceId(UUID(face_id_str))
        except ValueError as exc:
            raise ValidationError("face_id must be a valid UUID") from exc

        history = await self._service.get_face_history(face_id)
        return FaceHistoryResponse(
            face_id=face_id_str,
            history=[
                HistoryEntry(
                    process_id=entry["process_id"],
                    timestamp=entry["timestamp"],
                    process_type=entry.get("process_type"),
                    status=entry.get("status"),
                )
                for entry in history
            ],
        )

    async def get_process(self, process_id_str: str) -> ProcessResponse | None:
        try:
            process_id = ProcessId(UUID(process_id_str))
        except ValueError as exc:
            raise ValidationError("process_id must be a valid UUID") from exc

        process = await self._service.get_process(process_id)
        if process is None:
            return None
        return self._to_process_response(process)

    def _to_face_response(self, item: RecognitionResultItem) -> FaceResponse:
        return FaceResponse(
            face_id=str(item.face_id),
            status=item.status,
            name=item.name,
            metadata=item.metadata if item.metadata else None,
            bounding_box=BoundingBoxSchema(
                x=item.bounding_box.x,
                y=item.bounding_box.y,
                width=item.bounding_box.width,
                height=item.bounding_box.height,
            ),
            confidence=item.confidence,
        )

    def _to_process_response(self, process: ProcessRecord) -> ProcessResponse:
        return ProcessResponse(
            process_id=str(process.process_id),
            process_type=process.process_type,
            status=process.status,
            face_count=process.face_count,
            error_code=process.error_code,
            details=dict(process.details) if process.details else None,
            created_at=self._format_dt(process.created_at),
            completed_at=self._format_dt(process.completed_at) if process.completed_at else None,
        )

    @staticmethod
    def _format_dt(value: datetime | None) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
