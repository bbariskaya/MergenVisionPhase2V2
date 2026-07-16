"""Identity storage lifecycle orchestration service."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.application.ports.object_store import ObjectStore
from app.application.ports.unit_of_work import UnitOfWork
from app.application.ports.vector_store import VectorStore
from app.domain.entities.face_identity import FaceIdentity
from app.domain.entities.face_sample import FaceSample
from app.domain.entities.process_record import ProcessRecord
from app.domain.entities.recognition_result import RecognitionResult
from app.domain.errors import (
    IdentityResolutionError,
    ValidationError,
)
from app.domain.value_objects import BoundingBox, FaceId, ProcessId, ResultId, SampleId
from app.infrastructure.uuid7 import generate_uuid7

DIMENSION = 512
BUCKET_NAME = "mergenvision-face-samples"


@dataclass(frozen=True)
class RecognitionOutcome:
    process_id: ProcessId
    face_id: FaceId
    sample_id: SampleId | None
    status: str  # "known" | "anonymous" | "new_anonymous"
    bounding_box: BoundingBox
    match_confidence: float


class IdentityStorageLifecycleService:
    def __init__(
        self,
        unit_of_work: UnitOfWork,
        object_store: ObjectStore,
        vector_store: VectorStore,
    ) -> None:
        self._uow = unit_of_work
        self._object_store = object_store
        self._vector_store = vector_store

    def _validate_embedding(self, embedding: Sequence[float]) -> None:
        if len(embedding) != DIMENSION:
            raise ValidationError(f"Embedding must have length {DIMENSION}")
        if not all(math.isfinite(v) for v in embedding):
            raise ValidationError("Embedding values must be finite")
        norm = math.sqrt(sum(v * v for v in embedding))
        if norm == 0.0:
            raise ValidationError("Embedding norm must be non-zero")

    def _object_key(self, face_id: FaceId, sample_id: SampleId) -> str:
        return f"faces/{face_id}/{sample_id}/aligned.webp"

    async def resolve_or_create(
        self,
        crop_bytes: bytes,
        embedding: Sequence[float],
        bbox: BoundingBox,
        match_threshold: float,
    ) -> RecognitionOutcome:
        self._validate_embedding(embedding)

        process_id = ProcessId(generate_uuid7())
        process = ProcessRecord(
            process_id=process_id,
            process_type="image_recognize",
            status="processing",
        )
        async with self._uow:
            await self._uow.processes.add(process)
            await self._uow.commit()

        candidates = await self._vector_store.query(embedding, top_k=1)

        for candidate in candidates:
            if candidate.score < match_threshold:
                continue
            outcome = await self._accept_candidate(
                process_id=process_id,
                candidate=candidate,
                bbox=bbox,
                match_confidence=candidate.score,
            )
            if outcome is not None:
                return outcome

        highest_rejected = 0.0
        if candidates:
            highest_rejected = max(c.score for c in candidates if c.score < match_threshold)

        return await self._create_new_identity(
            process_id=process_id,
            crop_bytes=crop_bytes,
            embedding=embedding,
            bbox=bbox,
            match_confidence=highest_rejected,
        )

    async def _accept_candidate(
        self,
        process_id: ProcessId,
        candidate: Any,
        bbox: BoundingBox,
        match_confidence: float,
    ) -> RecognitionOutcome | None:
        async with self._uow:
            identity = await self._uow.face_identities.get_by_id(candidate.face_id)
            sample = await self._uow.face_samples.get_by_id(candidate.sample_id)
            if identity is None or sample is None:
                return None
            if not identity.is_active or not sample.is_active or sample.state != "active":
                return None

            status = identity.status
            result = RecognitionResult(
                result_id=ResultId(generate_uuid7()),
                process_id=process_id,
                face_id=identity.face_id,
                sample_id=sample.sample_id,
                status=status,
                bounding_box=bbox,
                match_confidence=match_confidence,
            )
            process = await self._uow.processes.get_by_id(process_id)
            assert process is not None
            process.complete(face_count=1)

            await self._uow.recognition_results.add(result)
            await self._uow.processes.update(process)
            await self._uow.commit()

            return RecognitionOutcome(
                process_id=process_id,
                face_id=identity.face_id,
                sample_id=sample.sample_id,
                status=status,
                bounding_box=bbox,
                match_confidence=match_confidence,
            )

    async def _create_new_identity(
        self,
        process_id: ProcessId,
        crop_bytes: bytes,
        embedding: Sequence[float],
        bbox: BoundingBox,
        match_confidence: float,
    ) -> RecognitionOutcome:
        face_id = FaceId(generate_uuid7())
        sample_id = SampleId(generate_uuid7())
        identity = FaceIdentity(face_id=face_id)
        sample = FaceSample(sample_id=sample_id, face_id=face_id)
        object_key = self._object_key(face_id, sample_id)

        async with self._uow:
            await self._uow.face_identities.add(identity)
            await self._uow.face_samples.add(sample)
            await self._uow.commit()

        try:
            await self._object_store.upload(object_key, crop_bytes, "image/webp")
        except Exception as exc:
            await self._persist_resolution_failure(
                face_id=face_id,
                sample_id=sample_id,
                process_id=process_id,
                error_code="minio_upload_failed",
                deactivate_identity=True,
            )
            raise IdentityResolutionError(f"MinIO upload failed: {exc}") from exc

        try:
            await self._vector_store.upsert(sample_id, face_id, embedding)
        except Exception as exc:
            await self._object_store.delete(object_key)
            await self._persist_resolution_failure(
                face_id=face_id,
                sample_id=sample_id,
                process_id=process_id,
                error_code="qdrant_upsert_failed",
                deactivate_identity=True,
            )
            raise IdentityResolutionError(f"Qdrant upsert failed: {exc}") from exc

        async with self._uow:
            sample_loaded = await self._uow.face_samples.get_by_id(sample_id)
            assert sample_loaded is not None
            sample_loaded.mark_active(BUCKET_NAME, object_key)
            await self._uow.face_samples.update(sample_loaded)

            result = RecognitionResult(
                result_id=ResultId(generate_uuid7()),
                process_id=process_id,
                face_id=face_id,
                sample_id=sample_id,
                status="new_anonymous",
                bounding_box=bbox,
                match_confidence=match_confidence,
            )
            await self._uow.recognition_results.add(result)

            process = await self._uow.processes.get_by_id(process_id)
            assert process is not None
            process.complete(face_count=1)
            await self._uow.processes.update(process)

            await self._uow.commit()

        return RecognitionOutcome(
            process_id=process_id,
            face_id=face_id,
            sample_id=sample_id,
            status="new_anonymous",
            bounding_box=bbox,
            match_confidence=match_confidence,
        )

    async def _persist_resolution_failure(
        self,
        face_id: FaceId,
        sample_id: SampleId,
        process_id: ProcessId,
        error_code: str,
        deactivate_identity: bool,
    ) -> None:
        async with self._uow:
            if deactivate_identity:
                identity = await self._uow.face_identities.get_by_id(face_id)
                if identity is not None and identity.is_active:
                    identity.deactivate()
                    await self._uow.face_identities.update(identity)

            sample = await self._uow.face_samples.get_by_id(sample_id)
            if sample is not None:
                sample.mark_failed(error_code)
                await self._uow.face_samples.update(sample)

            process = await self._uow.processes.get_by_id(process_id)
            if process is not None:
                process.fail(error_code)
                await self._uow.processes.update(process)

            await self._uow.commit()

    async def add_sample(
        self,
        face_id: FaceId,
        crop_bytes: bytes,
        embedding: Sequence[float],
    ) -> FaceSample:
        self._validate_embedding(embedding)

        async with self._uow:
            identity = await self._uow.face_identities.get_by_id(face_id)
            if identity is None or not identity.is_active:
                raise ValidationError("Face identity is not active")

        sample_id = SampleId(generate_uuid7())
        sample = FaceSample(sample_id=sample_id, face_id=face_id)
        object_key = self._object_key(face_id, sample_id)

        async with self._uow:
            await self._uow.face_samples.add(sample)
            await self._uow.commit()

        await self._object_store.upload(object_key, crop_bytes, "image/webp")
        await self._vector_store.upsert(sample_id, face_id, embedding)

        async with self._uow:
            loaded = await self._uow.face_samples.get_by_id(sample_id)
            assert loaded is not None
            loaded.mark_active(BUCKET_NAME, object_key)
            await self._uow.face_samples.update(loaded)
            await self._uow.commit()

        return loaded

    async def enroll_identity(
        self,
        face_id: FaceId,
        display_name: str,
        metadata: dict[str, Any],
    ) -> FaceIdentity:
        process_id = ProcessId(generate_uuid7())
        process = ProcessRecord(
            process_id=process_id,
            process_type="face_enroll",
            status="processing",
        )

        async with self._uow:
            await self._uow.processes.add(process)

            identity = await self._uow.face_identities.get_by_id(face_id)
            if identity is None:
                raise ValidationError("Face identity not found")
            if not identity.is_active:
                raise ValidationError("Face identity is not active")
            if identity.status != "anonymous":
                raise ValidationError("Only anonymous identities can be enrolled")

            identity.promote_to_known(display_name, metadata)
            await self._uow.face_identities.update(identity)

            process.complete(face_count=1)
            await self._uow.processes.update(process)

            await self._uow.commit()

        return identity

    async def deactivate_identity(self, face_id: FaceId) -> None:
        process_id = ProcessId(generate_uuid7())
        process = ProcessRecord(
            process_id=process_id,
            process_type="face_delete",
            status="processing",
        )

        async with self._uow:
            await self._uow.processes.add(process)

            identity = await self._uow.face_identities.get_by_id(face_id)
            if identity is None:
                raise ValidationError("Face identity not found")

            identity.deactivate()
            await self._uow.face_identities.update(identity)

            samples = await self._uow.face_samples.list_active_by_face_id(face_id)
            for sample in samples:
                sample.mark_inactive()
                await self._uow.face_samples.update(sample)

            process.complete(face_count=0)
            await self._uow.processes.update(process)

            await self._uow.commit()

        for sample in samples:
            with contextlib.suppress(Exception):
                await self._vector_store.set_active(sample.sample_id, False)
