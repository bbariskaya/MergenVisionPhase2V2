"""Identity storage lifecycle orchestration service."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.application.ports.id_generator import IdGenerator
from app.application.ports.object_store import ObjectStore
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.ports.vector_store import VectorCandidate, VectorStore
from app.domain.entities.face_identity import FaceIdentity
from app.domain.entities.face_sample import FaceSample
from app.domain.entities.process_record import ProcessRecord
from app.domain.entities.recognition_result import RecognitionResult
from app.domain.errors import (
    ConcurrentUpdateError,
    DomainError,
    IdentityResolutionError,
    ValidationError,
)
from app.domain.value_objects import BoundingBox, FaceId, ProcessId, ResultId, SampleId

logger = logging.getLogger(__name__)

DIMENSION = 512


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
        unit_of_work_factory: UnitOfWorkFactory,
        object_store: ObjectStore,
        vector_store: VectorStore,
        id_generator: IdGenerator,
        candidate_limit: int = 10,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._object_store = object_store
        self._vector_store = vector_store
        self._id_generator = id_generator
        self._candidate_limit = self._validate_candidate_limit(candidate_limit)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_candidate_limit(self, value: int) -> int:
        if not isinstance(value, int) or value < 1 or value > 100:
            raise ValidationError("candidate_limit must be an integer between 1 and 100")
        return value

    def _validate_threshold(self, threshold: float) -> None:
        if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValidationError("match_threshold must be finite and in [0.0, 1.0]")

    def _validate_crop_bytes(self, crop_bytes: bytes) -> None:
        if not isinstance(crop_bytes, bytes) or len(crop_bytes) == 0:
            raise ValidationError("crop_bytes must be non-empty bytes")

    def _validate_embedding(self, embedding: Sequence[float]) -> None:
        if len(embedding) != DIMENSION:
            raise ValidationError(f"Embedding must have length {DIMENSION}")
        if not all(math.isfinite(v) for v in embedding):
            raise ValidationError("Embedding values must be finite")
        norm = math.sqrt(sum(v * v for v in embedding))
        if norm == 0.0:
            raise ValidationError("Embedding norm must be non-zero")

    def _validate_bounding_box(self, bbox: BoundingBox) -> None:
        if bbox.x < 0 or bbox.y < 0:
            raise ValidationError("BoundingBox x and y must be non-negative")
        if bbox.width <= 0 or bbox.height <= 0:
            raise ValidationError("BoundingBox width and height must be positive")

    def _to_match_confidence(self, raw_score: float) -> float:
        if not math.isfinite(raw_score):
            return 0.0
        return min(1.0, max(0.0, raw_score))

    # ------------------------------------------------------------------
    # ID helpers
    # ------------------------------------------------------------------
    def _new_process_id(self) -> ProcessId:
        return ProcessId(self._id_generator.new_uuid7())

    def _new_face_id(self) -> FaceId:
        return FaceId(self._id_generator.new_uuid7())

    def _new_sample_id(self) -> SampleId:
        return SampleId(self._id_generator.new_uuid7())

    def _new_result_id(self) -> ResultId:
        return ResultId(self._id_generator.new_uuid7())

    def _object_key(self, face_id: FaceId, sample_id: SampleId) -> str:
        return f"faces/{face_id}/{sample_id}/aligned.webp"

    # ------------------------------------------------------------------
    # Process lifecycle helpers
    # ------------------------------------------------------------------
    async def start_process(self, process_type: str, details: dict[str, Any] | None = None) -> ProcessRecord:
        process_id = self._new_process_id()
        process = ProcessRecord(
            process_id=process_id,
            process_type=process_type,
            status="processing",
            details=details or {},
        )
        async with self._unit_of_work_factory() as uow:
            await uow.processes.add(process)
            await uow.commit()
        return process

    async def complete_process(
        self,
        process_id: ProcessId,
        face_count: int,
        details: dict[str, Any] | None = None,
    ) -> ProcessRecord:
        async with self._unit_of_work_factory() as uow:
            process = await uow.processes.get_by_id(process_id)
            if process is None:
                raise IdentityResolutionError("Process disappeared before completion")
            process.complete(face_count=face_count, details=details)
            await uow.processes.update(process)
            await uow.commit()
            return process

    async def fail_process(
        self,
        process_id: ProcessId,
        error_code: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        async with self._unit_of_work_factory() as uow:
            process = await uow.processes.get_by_id(process_id)
            if process is not None and process.status == "processing":
                process.fail(error_code, details)
                await uow.processes.update(process)
                await uow.commit()

    # ------------------------------------------------------------------
    # Public API - image recognition
    # ------------------------------------------------------------------
    async def resolve_or_create(
        self,
        crop_bytes: bytes,
        embedding: Sequence[float],
        bbox: BoundingBox,
        match_threshold: float,
    ) -> RecognitionOutcome:
        self._validate_crop_bytes(crop_bytes)
        self._validate_embedding(embedding)
        self._validate_bounding_box(bbox)
        self._validate_threshold(match_threshold)

        process_id = self._new_process_id()
        process = ProcessRecord(
            process_id=process_id,
            process_type="image_recognize",
            status="processing",
        )
        async with self._unit_of_work_factory() as uow:
            await uow.processes.add(process)
            await uow.commit()

        return await self._resolve_or_create_core(
            process_id=process_id,
            crop_bytes=crop_bytes,
            embedding=embedding,
            bbox=bbox,
            match_threshold=match_threshold,
            complete_process=True,
        )

    async def resolve_or_create_for_process(
        self,
        process_id: ProcessId,
        crop_bytes: bytes,
        embedding: Sequence[float],
        bbox: BoundingBox,
        match_threshold: float,
    ) -> RecognitionOutcome:
        self._validate_crop_bytes(crop_bytes)
        self._validate_embedding(embedding)
        self._validate_bounding_box(bbox)
        self._validate_threshold(match_threshold)

        return await self._resolve_or_create_core(
            process_id=process_id,
            crop_bytes=crop_bytes,
            embedding=embedding,
            bbox=bbox,
            match_threshold=match_threshold,
            complete_process=False,
        )

    async def add_sample(
        self,
        face_id: FaceId,
        crop_bytes: bytes,
        embedding: Sequence[float],
    ) -> FaceSample:
        self._validate_crop_bytes(crop_bytes)
        self._validate_embedding(embedding)

        async with self._unit_of_work_factory() as uow:
            identity = await uow.face_identities.get_by_id(face_id)
            if identity is None or not identity.is_active:
                raise ValidationError("Face identity is not active")

        sample_id = self._new_sample_id()
        sample = FaceSample(sample_id=sample_id, face_id=face_id)
        object_key = self._object_key(face_id, sample_id)

        async with self._unit_of_work_factory() as uow:
            await uow.face_samples.add(sample)
            await uow.commit()

        try:
            stat = await self._object_store.upload(object_key, crop_bytes, "image/webp")
        except Exception as exc:
            await self._fail_sample(sample_id, "minio_upload_failed")
            raise IdentityResolutionError(f"MinIO upload failed: {exc}") from exc

        try:
            await self._vector_store.upsert(sample_id, face_id, embedding)
        except Exception as exc:
            await self._delete_object_best_effort(object_key)
            await self._fail_sample(sample_id, "qdrant_upsert_failed")
            raise IdentityResolutionError(f"Qdrant upsert failed: {exc}") from exc

        try:
            async with self._unit_of_work_factory() as uow:
                loaded = await uow.face_samples.get_by_id(sample_id)
                if loaded is None:
                    raise IdentityResolutionError("Sample disappeared before activation")
                loaded.mark_active(stat.bucket, stat.key)
                await uow.face_samples.update(loaded)
                await uow.commit()
                return loaded
        except Exception as exc:
            await self._vector_store.delete(sample_id)
            await self._delete_object_best_effort(object_key)
            await self._fail_sample(sample_id, "activation_failed")
            raise IdentityResolutionError(f"Sample activation failed: {exc}") from exc

    async def enroll_identity(
        self,
        face_id: FaceId,
        display_name: str,
        metadata: dict[str, Any],
    ) -> FaceIdentity:
        process_id = self._new_process_id()
        process = ProcessRecord(
            process_id=process_id,
            process_type="face_enroll",
            status="processing",
        )

        async with self._unit_of_work_factory() as uow:
            await uow.processes.add(process)
            await uow.commit()

        try:
            async with self._unit_of_work_factory() as uow:
                identity = await uow.face_identities.get_by_id(face_id)
                if identity is None:
                    raise ValidationError("Face identity not found")
                if not identity.is_active:
                    raise ValidationError("Face identity is not active")
                if identity.status != "anonymous":
                    raise ValidationError("Only anonymous identities can be enrolled")

                expected_version = identity.version
                identity.promote_to_known(display_name, metadata)
                updated = await uow.face_identities.update_with_expected_version(
                    identity,
                    expected_version,
                )

                process_loaded = await uow.processes.get_by_id(process_id)
                if process_loaded is None:
                    raise IdentityResolutionError("Process disappeared during enrollment")
                process_loaded.complete(face_count=1)
                await uow.processes.update(process_loaded)
                await uow.commit()
                return updated
        except Exception as exc:
            await self._fail_process(process_id, "enrollment_failed")
            if isinstance(exc, ValidationError | ConcurrentUpdateError):
                raise
            raise IdentityResolutionError(f"Enrollment failed: {exc}") from exc

    async def deactivate_identity(self, face_id: FaceId) -> None:
        process_id = self._new_process_id()
        process = ProcessRecord(
            process_id=process_id,
            process_type="face_delete",
            status="processing",
        )

        async with self._unit_of_work_factory() as uow:
            await uow.processes.add(process)
            await uow.commit()

        try:
            async with self._unit_of_work_factory() as uow:
                identity = await uow.face_identities.get_by_id(face_id)
                if identity is None:
                    raise ValidationError("Face identity not found")

                expected_version = identity.version
                identity.deactivate()
                await uow.face_identities.update_with_expected_version(
                    identity,
                    expected_version,
                )

                samples = await uow.face_samples.list_active_by_face_id(face_id)
                for sample in samples:
                    sample.mark_inactive()
                    await uow.face_samples.update(sample)

                process_loaded = await uow.processes.get_by_id(process_id)
                if process_loaded is None:
                    raise IdentityResolutionError("Process disappeared during deactivation")
                process_loaded.complete(face_count=0)
                await uow.processes.update(process_loaded)
                await uow.commit()
        except Exception as exc:
            await self._fail_process(process_id, "deactivation_failed")
            if isinstance(exc, ValidationError | ConcurrentUpdateError):
                raise
            raise IdentityResolutionError(f"Deactivation failed: {exc}") from exc

        # Best-effort Qdrant deactivation after PG commit.
        for sample in samples:
            try:
                await self._vector_store.set_active(sample.sample_id, False)
            except Exception as exc:
                logger.warning(
                    "Qdrant deactivation cleanup warning for sample %s: %s",
                    sample.sample_id,
                    exc,
                )

    # ------------------------------------------------------------------
    # Core resolution (shared)
    # ------------------------------------------------------------------
    async def _resolve_or_create_core(
        self,
        process_id: ProcessId,
        crop_bytes: bytes,
        embedding: Sequence[float],
        bbox: BoundingBox,
        match_threshold: float,
        complete_process: bool,
    ) -> RecognitionOutcome:
        try:
            candidates = await self._vector_store.query(embedding, top_k=self._candidate_limit)
        except Exception as exc:
            if complete_process:
                await self._fail_process(process_id, "vector_query_failed")
            raise IdentityResolutionError("Vector query failed during identity resolution") from exc

        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)

        for candidate in sorted_candidates:
            if not math.isfinite(candidate.score):
                continue
            if candidate.score < match_threshold:
                continue
            outcome = await self._try_accept_candidate(
                process_id=process_id,
                candidate=candidate,
                bbox=bbox,
                complete_process=complete_process,
            )
            if outcome is not None:
                return outcome

        rejected_scores = [
            c.score
            for c in sorted_candidates
            if math.isfinite(c.score) and c.score < match_threshold
        ]
        highest_rejected = self._to_match_confidence(max(rejected_scores, default=0.0))

        return await self._create_new_identity(
            process_id=process_id,
            crop_bytes=crop_bytes,
            embedding=embedding,
            bbox=bbox,
            match_confidence=highest_rejected,
            complete_process=complete_process,
        )

    # ------------------------------------------------------------------
    # Candidate acceptance
    # ------------------------------------------------------------------
    async def _try_accept_candidate(
        self,
        process_id: ProcessId,
        candidate: VectorCandidate,
        bbox: BoundingBox,
        complete_process: bool,
    ) -> RecognitionOutcome | None:
        try:
            async with self._unit_of_work_factory() as uow:
                identity = await uow.face_identities.get_by_id(candidate.face_id)
                sample = await uow.face_samples.get_by_id(candidate.sample_id)

                if identity is None or sample is None:
                    return None
                if not identity.is_active or identity.deleted_at is not None:
                    return None
                if sample.state != "active" or not sample.is_active:
                    return None
                if sample.face_id != identity.face_id:
                    return None

                status = identity.status
                confidence = self._to_match_confidence(candidate.score)
                result = RecognitionResult(
                    result_id=self._new_result_id(),
                    process_id=process_id,
                    face_id=identity.face_id,
                    sample_id=sample.sample_id,
                    status=status,
                    bounding_box=bbox,
                    match_confidence=confidence,
                )

                process = await uow.processes.get_by_id(process_id)
                if process is None:
                    raise IdentityResolutionError("Process disappeared during candidate acceptance")
                if complete_process:
                    process.complete(face_count=1)

                await uow.recognition_results.add(result)
                await uow.processes.update(process)
                await uow.commit()

                return RecognitionOutcome(
                    process_id=process_id,
                    face_id=identity.face_id,
                    sample_id=sample.sample_id,
                    status=status,
                    bounding_box=bbox,
                    match_confidence=confidence,
                )
        except Exception as exc:
            if complete_process:
                await self._fail_process(process_id, "candidate_acceptance_failed")
            if isinstance(exc, DomainError):
                raise
            raise IdentityResolutionError("Failed to accept recognition candidate") from exc

    # ------------------------------------------------------------------
    # New identity creation with compensation
    # ------------------------------------------------------------------
    async def _create_new_identity(
        self,
        process_id: ProcessId,
        crop_bytes: bytes,
        embedding: Sequence[float],
        bbox: BoundingBox,
        match_confidence: float,
        complete_process: bool,
    ) -> RecognitionOutcome:
        face_id = self._new_face_id()
        sample_id = self._new_sample_id()
        identity = FaceIdentity(face_id=face_id)
        sample = FaceSample(sample_id=sample_id, face_id=face_id)
        object_key = self._object_key(face_id, sample_id)

        async with self._unit_of_work_factory() as uow:
            await uow.face_identities.add(identity)
            await uow.face_samples.add(sample)
            await uow.commit()

        try:
            stat = await self._object_store.upload(object_key, crop_bytes, "image/webp")
        except Exception as exc:
            await self._persist_resolution_failure(
                face_id=face_id,
                sample_id=sample_id,
                process_id=process_id,
                error_code="minio_upload_failed",
            )
            raise IdentityResolutionError(f"MinIO upload failed: {exc}") from exc

        try:
            await self._vector_store.upsert(sample_id, face_id, embedding)
        except Exception as exc:
            await self._delete_object_best_effort(object_key)
            await self._persist_resolution_failure(
                face_id=face_id,
                sample_id=sample_id,
                process_id=process_id,
                error_code="qdrant_upsert_failed",
            )
            raise IdentityResolutionError(f"Qdrant upsert failed: {exc}") from exc

        try:
            async with self._unit_of_work_factory() as uow:
                loaded_sample = await uow.face_samples.get_by_id(sample_id)
                if loaded_sample is None:
                    raise IdentityResolutionError("Sample disappeared before activation")
                loaded_sample.mark_active(stat.bucket, stat.key)
                await uow.face_samples.update(loaded_sample)

                result = RecognitionResult(
                    result_id=self._new_result_id(),
                    process_id=process_id,
                    face_id=face_id,
                    sample_id=sample_id,
                    status="new_anonymous",
                    bounding_box=bbox,
                    match_confidence=match_confidence,
                )
                await uow.recognition_results.add(result)

                process = await uow.processes.get_by_id(process_id)
                if process is None:
                    raise IdentityResolutionError("Process disappeared before completion")
                if complete_process:
                    process.complete(face_count=1)
                await uow.processes.update(process)
                await uow.commit()
        except Exception as exc:
            await self._vector_store.delete(sample_id)
            await self._delete_object_best_effort(object_key)
            await self._persist_resolution_failure(
                face_id=face_id,
                sample_id=sample_id,
                process_id=process_id,
                error_code="finalization_failed",
            )
            raise IdentityResolutionError(f"Identity finalization failed: {exc}") from exc

        return RecognitionOutcome(
            process_id=process_id,
            face_id=face_id,
            sample_id=sample_id,
            status="new_anonymous",
            bounding_box=bbox,
            match_confidence=match_confidence,
        )

    # ------------------------------------------------------------------
    # Failure persistence helpers
    # ------------------------------------------------------------------
    async def _persist_resolution_failure(
        self,
        face_id: FaceId,
        sample_id: SampleId,
        process_id: ProcessId,
        error_code: str,
    ) -> None:
        async with self._unit_of_work_factory() as uow:
            identity = await uow.face_identities.get_by_id(face_id)
            if identity is not None and identity.is_active:
                identity.deactivate()
                await uow.face_identities.update(identity)

            sample = await uow.face_samples.get_by_id(sample_id)
            if sample is not None:
                sample.mark_failed(error_code)
                await uow.face_samples.update(sample)

            process = await uow.processes.get_by_id(process_id)
            if process is not None:
                process.fail(error_code)
                await uow.processes.update(process)

            await uow.commit()

    async def _fail_process(self, process_id: ProcessId, error_code: str) -> None:
        await self.fail_process(process_id, error_code)

    async def _fail_sample(self, sample_id: SampleId, error_code: str) -> None:
        async with self._unit_of_work_factory() as uow:
            sample = await uow.face_samples.get_by_id(sample_id)
            if sample is not None and sample.state == "pending":
                sample.mark_failed(error_code)
                await uow.face_samples.update(sample)
                await uow.commit()

    async def _delete_object_best_effort(self, object_key: str) -> None:
        try:
            await self._object_store.delete(object_key)
        except Exception as exc:
            logger.warning("MinIO cleanup warning for %s: %s", object_key, exc)
