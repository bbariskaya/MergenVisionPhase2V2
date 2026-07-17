"""Phase 2 Milestone 0.3 — image recognition guarded lifecycle tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pytest

from app.application.ports.image_recognition import (
    ImageRecognitionEngine,
    NativeFaceDetection,
    NativeRecognitionResult,
)
from app.application.services.identity_storage_lifecycle_service import (
    RecognitionOutcome,
)
from app.application.services.image_recognition_service import (
    ImageRecognitionService,
)
from app.domain.entities.process_record import ProcessRecord
from app.domain.errors import IdentityResolutionError
from app.domain.value_objects import BoundingBox, FaceId, ProcessId, SampleId


@dataclass
class _FakeDetection:
    bbox: BoundingBox
    confidence: float = 0.9
    embedding: list[float] = field(default_factory=lambda: [0.1] * 512)
    crop_bytes: bytes = b"fake-crop"

    def to_native(self) -> NativeFaceDetection:
        return NativeFaceDetection(
            bounding_box=self.bbox,
            detector_confidence=self.confidence,
            embedding=list(self.embedding),
            aligned_crop_bytes=self.crop_bytes,
        )


@dataclass
class _FakeEngine(ImageRecognitionEngine):
    detections: list[_FakeDetection]
    should_fail: bool = False

    async def detect_and_embed(self, image_bytes: bytes) -> NativeRecognitionResult:
        if self.should_fail:
            raise RuntimeError("detector failed")
        return NativeRecognitionResult(
            image_width=640,
            image_height=480,
            detections=[d.to_native() for d in self.detections],
        )


@dataclass
class _FakeLifecycle:
    outcomes: list[RecognitionOutcome] = field(default_factory=list)
    failed_processes: list[tuple[ProcessId, str, dict[str, Any] | None]] = field(
        default_factory=list
    )
    fail_after_index: int | None = None

    async def start_process(self, process_type: str, details: dict[str, Any] | None = None) -> ProcessRecord:
        return ProcessRecord(
            process_id=ProcessId(uuid4()),
            process_type=process_type,
            status="processing",
            details=details or {},
        )

    async def complete_process(
        self,
        process_id: ProcessId,
        face_count: int,
        details: dict[str, Any] | None = None,
    ) -> ProcessRecord:
        return ProcessRecord(
            process_id=process_id,
            process_type="image_recognize",
            status="completed",
            face_count=face_count,
            details=details or {},
        )

    async def fail_process(
        self,
        process_id: ProcessId,
        error_code: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.failed_processes.append((process_id, error_code, details))

    async def resolve_or_create_for_process(
        self,
        process_id: ProcessId,
        crop_bytes: bytes,
        embedding: list[float],
        bbox: BoundingBox,
        match_threshold: float,
    ) -> RecognitionOutcome:
        call_index = len(self.outcomes)
        self.outcomes.append(None)  # type: ignore[arg-type]
        if self.fail_after_index is not None and call_index >= self.fail_after_index:
            raise IdentityResolutionError("resolution failed")
        outcome = RecognitionOutcome(
            process_id=process_id,
            face_id=FaceId(uuid4()),
            sample_id=SampleId(uuid4()),
            status="new_anonymous",
            bounding_box=bbox,
            match_confidence=0.7,
        )
        self.outcomes[call_index] = outcome
        return outcome


def _service(lifecycle: _FakeLifecycle, engine: _FakeEngine) -> ImageRecognitionService:
    return ImageRecognitionService(
        lifecycle_service=lifecycle,  # type: ignore[arg-type]
        unit_of_work_factory=lambda: None,  # type: ignore[return-value]
        max_image_bytes=10_000_000,
        model_version="v1",
        engine=engine,  # type: ignore[arg-type]
        match_threshold=0.55,
    )


@pytest.mark.asyncio
async def test_native_inference_failure_marks_process_failed() -> None:
    lifecycle = _FakeLifecycle()
    engine = _FakeEngine(detections=[], should_fail=True)
    service = _service(lifecycle, engine)

    with pytest.raises(RuntimeError):
        await service.recognize_image(b"valid-jpeg-bytes")

    assert len(lifecycle.failed_processes) == 1
    _, error_code, _ = lifecycle.failed_processes[0]
    assert error_code == "native_inference_failed"


@pytest.mark.asyncio
async def test_partial_resolution_failure_marks_process_failed_and_preserves_results() -> None:
    lifecycle = _FakeLifecycle(fail_after_index=1)
    engine = _FakeEngine(
        detections=[
            _FakeDetection(bbox=BoundingBox(x=0, y=0, width=10, height=10)),
            _FakeDetection(bbox=BoundingBox(x=20, y=20, width=10, height=10)),
        ]
    )
    service = _service(lifecycle, engine)

    with pytest.raises(IdentityResolutionError):
        await service.recognize_image(b"valid-jpeg-bytes")

    assert len(lifecycle.failed_processes) == 1
    _, error_code, details = lifecycle.failed_processes[0]
    assert error_code == "identity_resolution_failed"
    assert details is not None
    assert details.get("partial_face_count") == 1
    # The first face was persisted by the fake lifecycle.
    assert len([o for o in lifecycle.outcomes if isinstance(o, RecognitionOutcome)]) == 1
