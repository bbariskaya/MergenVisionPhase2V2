"""Image recognition orchestration service.

This is the application/business layer that bridges the native GPU runtime
with the identity storage lifecycle. It decides *which* detected faces should
be resolved, feeds them to the identity lifecycle, and assembles API-facing
result DTOs.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC
from typing import Any

from app.application.ports.image_recognition import (
    ImageRecognitionEngine,
    NativeFaceDetection,
)
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
    RecognitionOutcome,
)
from app.domain.entities.face_identity import FaceIdentity
from app.domain.entities.process_record import ProcessRecord
from app.domain.errors import ValidationError
from app.domain.value_objects import BoundingBox, FaceId, ProcessId

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecognitionResultItem:
    face_id: FaceId
    status: str
    name: str | None
    metadata: dict[str, Any]
    bounding_box: BoundingBox
    confidence: float


@dataclass(frozen=True)
class RecognizeResult:
    process: ProcessRecord
    faces: list[RecognitionResultItem]


class ImageRecognitionService:
    def __init__(
        self,
        lifecycle_service: IdentityStorageLifecycleService,
        unit_of_work_factory: UnitOfWorkFactory,
        max_image_bytes: int,
        model_version: str,
        engine: ImageRecognitionEngine | None = None,
        engine_factory: Callable[[], ImageRecognitionEngine] | None = None,
        match_threshold: float = 0.6,
    ) -> None:
        if engine is None and engine_factory is None:
            raise ValueError("Provide either engine or engine_factory")
        if not isinstance(max_image_bytes, int) or max_image_bytes <= 0:
            raise ValueError("max_image_bytes must be a positive integer")
        if not model_version:
            raise ValueError("model_version must be provided")

        self._engine = engine
        self._engine_factory = engine_factory
        self._engine_lock = asyncio.Lock()
        self._lifecycle = lifecycle_service
        self._unit_of_work_factory = unit_of_work_factory
        self._max_image_bytes = max_image_bytes
        self._model_version = model_version
        self._match_threshold = self._validate_threshold(match_threshold)

    async def _get_engine(self) -> ImageRecognitionEngine:
        if self._engine is not None:
            return self._engine
        if self._engine_factory is None:
            raise RuntimeError("Image recognition engine is not configured")
        async with self._engine_lock:
            if self._engine is None:
                self._engine = self._engine_factory()
        return self._engine

    def _validate_threshold(self, threshold: float) -> float:
        if not isinstance(threshold, int | float):
            raise ValidationError("match_threshold must be a number")
        if threshold < 0.0 or threshold > 1.0:
            raise ValidationError("match_threshold must be in [0.0, 1.0]")
        return float(threshold)

    async def recognize_image(self, image_bytes: bytes) -> RecognizeResult:
        if not isinstance(image_bytes, bytes) or len(image_bytes) == 0:
            raise ValidationError("image_bytes must be non-empty bytes")
        if len(image_bytes) > self._max_image_bytes:
            raise ValidationError(
                f"Image exceeds maximum allowed size of {self._max_image_bytes} bytes"
            )

        process = await self._lifecycle.start_process(
            "image_recognize",
            {"model_version": self._model_version},
        )

        engine = await self._get_engine()
        try:
            raw = await engine.detect_and_embed(image_bytes)
        except Exception:
            await self._lifecycle.fail_process(process.process_id, "native_inference_failed")
            raise

        faces = await self._resolve_detections(process.process_id, raw.detections)
        details = {
            "model_version": self._model_version,
            "image_width": raw.image_width,
            "image_height": raw.image_height,
            "detection_count": len(raw.detections),
        }
        completed = await self._lifecycle.complete_process(
            process.process_id,
            face_count=len(faces),
            details=details,
        )
        return RecognizeResult(process=completed, faces=faces)

    async def _resolve_detections(
        self,
        process_id: ProcessId,
        detections: list[NativeFaceDetection],
    ) -> list[RecognitionResultItem]:
        outcomes: list[RecognitionOutcome] = []
        for det in detections:
            outcome = await self._lifecycle.resolve_or_create_for_process(
                process_id=process_id,
                crop_bytes=det.aligned_crop_bytes,
                embedding=det.embedding,
                bbox=det.bounding_box,
                match_threshold=self._match_threshold,
            )
            outcomes.append(outcome)

        identity_meta: dict[FaceId, tuple[str | None, dict[str, Any]]] = {}
        for outcome in outcomes:
            if outcome.status == "known" and outcome.face_id not in identity_meta:
                identity_meta[outcome.face_id] = await self._load_identity_meta(outcome.face_id)

        return [
            RecognitionResultItem(
                face_id=outcome.face_id,
                status=outcome.status,
                name=identity_meta.get(outcome.face_id, (None, {}))[0]
                if outcome.status == "known"
                else None,
                metadata=dict(identity_meta.get(outcome.face_id, (None, {}))[1])
                if outcome.status == "known"
                else {},
                bounding_box=outcome.bounding_box,
                confidence=outcome.match_confidence,
            )
            for outcome in outcomes
        ]

    async def _load_identity_meta(
        self,
        face_id: FaceId,
    ) -> tuple[str | None, dict[str, Any]]:
        async with self._unit_of_work_factory() as uow:
            identity = await uow.face_identities.get_by_id(face_id)
            if identity is None:
                return (None, {})
            return (identity.display_name, dict(identity.identity_metadata))

    async def enroll_face(
        self,
        face_id: FaceId,
        display_name: str,
        metadata: dict[str, Any] | None,
    ) -> FaceIdentity:
        return await self._lifecycle.enroll_identity(
            face_id=face_id,
            display_name=display_name,
            metadata=metadata or {},
        )

    async def get_identity_detail(self, face_id: FaceId) -> FaceIdentity | None:
        async with self._unit_of_work_factory() as uow:
            return await uow.face_identities.get_by_id(face_id)

    async def delete_identity(self, face_id: FaceId) -> bool:
        await self._lifecycle.deactivate_identity(face_id)
        return True

    async def get_face_history(
        self,
        face_id: FaceId,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        async with self._unit_of_work_factory() as uow:
            results = await uow.recognition_results.list_by_face_id(face_id, limit=limit)
            process_ids = [r.process_id for r in results]
            processes = {
                p.process_id: p
                for p in [await uow.processes.get_by_id(pid) for pid in process_ids]
                if p is not None
            }

        history: list[dict[str, Any]] = []
        for result in results:
            process = processes.get(result.process_id)
            history.append(
                {
                    "process_id": str(result.process_id),
                    "timestamp": result.created_at.replace(tzinfo=UTC).isoformat()
                    if result.created_at.tzinfo is None
                    else result.created_at.isoformat(),
                    "process_type": process.process_type if process else None,
                    "status": process.status if process else None,
                    "recognition_status": result.status,
                    "match_confidence": float(result.match_confidence),
                }
            )
        return history

    async def get_process(self, process_id: ProcessId) -> ProcessRecord | None:
        async with self._unit_of_work_factory() as uow:
            return await uow.processes.get_by_id(process_id)
