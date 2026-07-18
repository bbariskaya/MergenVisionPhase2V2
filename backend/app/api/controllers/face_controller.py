"""HTTP-facing controllers for the face recognition API.

Controllers map between HTTP request/response DTOs and the application service
layer. They contain no business rules beyond trivial validation/formatting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from app.api.schemas import (
    BoundingBoxSchema,
    EnrollResponse,
    FaceHistoryResponse,
    FaceResponse,
    FaceSampleResponse,
    FaceSamplesResponse,
    HistoryEntry,
    IdentityDetailResponse,
    IdentityListResponse,
    IdentitySummary,
    ProcessResponse,
    RecognizeResponse,
)
from app.application.ports.object_store import ObjectStore
from app.application.services.image_recognition_service import (
    ImageRecognitionService,
    RecognitionResultItem,
)
from app.domain.entities.face_sample import FaceSample
from app.domain.entities.process_record import ProcessRecord
from app.domain.errors import ValidationError
from app.domain.value_objects import FaceId, PersonId, ProcessId, SampleId


@dataclass(frozen=True)
class RecognizeRequestData:
    image_bytes: bytes
    filename: str | None


class FaceController:
    def __init__(
        self,
        service: ImageRecognitionService,
        object_store: ObjectStore | None = None,
    ) -> None:
        self._service = service
        self._object_store = object_store

    async def recognize(self, request_id: str, data: RecognizeRequestData) -> RecognizeResponse:
        result = await self._service.recognize_image(data.image_bytes)
        process = result.process
        return RecognizeResponse(
            request_id=request_id,
            process_id=str(process.process_id),
            status=process.status,
            face_count=len(result.faces),
            faces=[self._to_face_response(f) for f in result.faces],
        )

    async def enroll(
        self,
        request_id: str,
        face_id: str,
        display_name: str,
        metadata: dict[str, Any],
    ) -> EnrollResponse:
        try:
            parsed_face_id = FaceId(UUID(face_id))
        except ValueError as exc:
            raise ValidationError("face_id must be a valid UUID") from exc

        identity = await self._service.enroll_face(
            face_id=parsed_face_id,
            display_name=display_name,
            metadata=metadata or {},
        )
        # Enroll uses its own internal process; we surface a synthetic
        # processId correlation for the API envelope.
        process_id = str(identity.process_id) if hasattr(identity, "process_id") and identity.process_id else request_id
        return EnrollResponse(
            request_id=request_id,
            process_id=process_id,
            face_id=str(identity.face_id),
            status=identity.status,
            name=identity.display_name or display_name,
            metadata=dict(identity.identity_metadata),
            person_id=str(identity.person_id) if identity.person_id else None,
        )

    async def assign_to_person(
        self,
        request_id: str,
        face_id_str: str,
        person_id_str: str,
    ) -> EnrollResponse:
        try:
            face_id = FaceId(UUID(face_id_str))
            person_id = PersonId(UUID(person_id_str))
        except ValueError as exc:
            raise ValidationError("face_id and person_id must be valid UUIDs") from exc

        identity = await self._service.assign_face_to_person(face_id, person_id)
        return EnrollResponse(
            request_id=request_id,
            process_id=request_id,
            face_id=str(identity.face_id),
            status=identity.status,
            name=identity.display_name or "",
            metadata=dict(identity.identity_metadata),
            person_id=str(identity.person_id) if identity.person_id else None,
        )

    async def get_identity(self, request_id: str, face_id_str: str) -> IdentityDetailResponse | None:
        try:
            face_id = FaceId(UUID(face_id_str))
        except ValueError as exc:
            raise ValidationError("face_id must be a valid UUID") from exc

        identity = await self._service.get_identity_detail(face_id)
        if identity is None:
            return None
        return IdentityDetailResponse(
            request_id=request_id,
            face_id=str(identity.face_id),
            status=identity.status,
            name=identity.display_name,
            person_id=str(identity.person_id) if identity.person_id else None,
            metadata=dict(identity.identity_metadata),
            created_at=self._format_dt(identity.created_at),
            updated_at=self._format_dt(identity.updated_at),
        )

    async def delete_face_sample(
        self,
        request_id: str,
        face_id_str: str,
        sample_id_str: str,
    ) -> None:
        try:
            face_id = FaceId(UUID(face_id_str))
            sample_id = UUID(sample_id_str)
        except ValueError as exc:
            raise ValidationError("face_id and sample_id must be valid UUIDs") from exc

        await self._service.delete_face_sample(face_id, SampleId(sample_id))

    async def delete_identity(self, request_id: str, face_id_str: str) -> EnrollResponse:
        try:
            face_id = FaceId(UUID(face_id_str))
        except ValueError as exc:
            raise ValidationError("face_id must be a valid UUID") from exc

        await self._service.delete_identity(face_id)
        return EnrollResponse(
            request_id=request_id,
            process_id=request_id,
            face_id=face_id_str,
            status="deleted",
            name="",
            metadata={},
        )

    async def get_history(self, request_id: str, face_id_str: str) -> FaceHistoryResponse:
        try:
            face_id = FaceId(UUID(face_id_str))
        except ValueError as exc:
            raise ValidationError("face_id must be a valid UUID") from exc

        history = await self._service.get_face_history(face_id)
        return FaceHistoryResponse(
            request_id=request_id,
            face_id=face_id_str,
            history=[
                HistoryEntry(
                    process_id=entry["process_id"],
                    timestamp=entry["timestamp"],
                    process_type=entry.get("process_type"),
                    status=entry.get("status"),
                    recognition_status=entry.get("recognition_status"),
                    match_confidence=entry.get("match_confidence"),
                )
                for entry in history
            ],
        )

    async def get_process(self, request_id: str, process_id_str: str) -> ProcessResponse | None:
        try:
            process_id = ProcessId(UUID(process_id_str))
        except ValueError as exc:
            raise ValidationError("process_id must be a valid UUID") from exc

        process = await self._service.get_process(process_id)
        if process is None:
            return None
        return self._to_process_response(request_id, process)

    async def list_identities(
        self,
        request_id: str,
        query: str | None = None,
    ) -> IdentityListResponse:
        identities = await self._service.list_identities(query=query, status="known")
        return IdentityListResponse(
            request_id=request_id,
            count=len(identities),
            identities=[
                IdentitySummary(
                    face_id=str(identity.face_id),
                    status=identity.status,
                    name=identity.display_name,
                    metadata=dict(identity.identity_metadata),
                    created_at=self._format_dt(identity.created_at),
                    updated_at=self._format_dt(identity.updated_at),
                )
                for identity in identities
            ],
        )

    async def get_face_samples(
        self,
        request_id: str,
        face_id_str: str,
    ) -> FaceSamplesResponse | None:
        try:
            face_id = FaceId(UUID(face_id_str))
        except ValueError as exc:
            raise ValidationError("face_id must be a valid UUID") from exc

        identity = await self._service.get_identity_detail(face_id)
        if identity is None:
            return None

        samples = await self._service.list_face_samples(face_id)
        sample_responses: list[FaceSampleResponse] = []
        for sample in samples:
            image_url = (
                f"/api/v1/faces/{face_id_str}/samples/{sample.sample_id}/image"
                if sample.object_key
                else None
            )
            sample_responses.append(self._to_sample_response(sample, image_url))

        return FaceSamplesResponse(
            request_id=request_id,
            face_id=face_id_str,
            count=len(sample_responses),
            samples=sample_responses,
        )

    async def add_face_sample(
        self,
        request_id: str,
        face_id_str: str,
        image_bytes: bytes,
    ) -> FaceSampleResponse | None:
        try:
            face_id = FaceId(UUID(face_id_str))
        except ValueError as exc:
            raise ValidationError("face_id must be a valid UUID") from exc

        identity = await self._service.get_identity_detail(face_id)
        if identity is None:
            return None

        sample = await self._service.add_face_sample(face_id, image_bytes)
        image_url = f"/api/v1/faces/{face_id_str}/samples/{sample.sample_id}/image"
        return self._to_sample_response(sample, image_url)

    async def get_sample_image_bytes(
        self,
        face_id_str: str,
        sample_id_str: str,
    ) -> tuple[bytes, str] | None:
        try:
            face_id = FaceId(UUID(face_id_str))
            sample_id = UUID(sample_id_str)
        except ValueError as exc:
            raise ValidationError("face_id and sample_id must be valid UUIDs") from exc

        identity = await self._service.get_identity_detail(face_id)
        if identity is None:
            return None

        for sample in await self._service.list_face_samples(face_id):
            if sample.sample_id == sample_id:
                object_key = sample.object_key
                break
        else:
            return None

        if not object_key or self._object_store is None:
            return None

        data = await self._object_store.get(object_key)
        if data is None:
            return None

        content_type = "image/webp"
        if object_key.lower().endswith(".jpg") or object_key.lower().endswith(".jpeg"):
            content_type = "image/jpeg"
        elif object_key.lower().endswith(".png"):
            content_type = "image/png"
        return data, content_type

    def _to_sample_response(
        self,
        sample: FaceSample,
        image_url: str | None,
    ) -> FaceSampleResponse:
        return FaceSampleResponse(
            sample_id=str(sample.sample_id),
            face_id=str(sample.face_id),
            state=sample.state,
            image_url=image_url,
            created_at=self._format_dt(sample.created_at),
            activated_at=self._format_dt(sample.activated_at),
        )

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

    def _to_process_response(self, request_id: str, process: ProcessRecord) -> ProcessResponse:
        return ProcessResponse(
            request_id=request_id,
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
