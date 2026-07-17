"""Phase 2 Milestone 0.5 — input validation tests."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.api.controllers.face_controller import FaceController
from app.api.main import create_app
from app.api.routes.dependencies import get_face_controller
from app.application.services.image_recognition_service import RecognizeResult
from app.domain.entities.process_record import ProcessRecord
from app.domain.value_objects import BoundingBox, FaceId, ProcessId


@dataclass(frozen=True)
class _FakeDetection:
    bbox: BoundingBox
    confidence: float


class _FakeValidationService:
    async def recognize_image(self, image_bytes: bytes) -> RecognizeResult:
        process = ProcessRecord(
            process_id=ProcessId(UUID("018f0000-0000-7b0e-8000-000000000001")),
            process_type="image_recognize",
            status="completed",
            face_count=0,
        )
        return RecognizeResult(process=process, faces=[])

    async def enroll_face(
        self, face_id: FaceId, display_name: str, metadata: dict[str, Any]
    ) -> Any:
        class _Identity:
            def __init__(self, face_id: FaceId, display_name: str) -> None:
                self.face_id = face_id
                self.status = "known"
                self.display_name = display_name
                self.identity_metadata = {}
        return _Identity(face_id, display_name)

    async def get_identity_detail(self, face_id: FaceId) -> Any:
        return None

    async def delete_identity(self, face_id: FaceId) -> bool:
        return True

    async def get_face_history(self, face_id: FaceId, limit: int = 100) -> list[dict[str, Any]]:
        return []

    async def get_process(self, process_id: ProcessId) -> Any:
        return None


def _client() -> TestClient:
    app = create_app()
    fake_controller = FaceController(_FakeValidationService())
    app.dependency_overrides[get_face_controller] = lambda: fake_controller
    app.state.face_controller = fake_controller
    return TestClient(app)


def _make_jpeg_header(width: int, height: int) -> bytes:
    """Minimal JPEG byte sequence with SOF0 header (no scan data)."""
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        + b"\xff\xc0\x00\x0b\x08\x00\x01"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x01\x01\x11\x00"
    )


def test_empty_file_is_rejected() -> None:
    with _client() as client:
        response = client.post(
            "/api/v1/faces/recognize",
            files={"image": ("empty.jpg", BytesIO(b""), "image/jpeg")},
        )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "INVALID_MEDIA"
    assert "requestId" in body


def test_non_jpeg_magic_returns_415() -> None:
    with _client() as client:
        response = client.post(
            "/api/v1/faces/recognize",
            files={"image": ("not.jpg", BytesIO(b"\x89PNG\r\n\x1a\n"), "image/jpeg")},
        )
    assert response.status_code == 415
    body = response.json()
    assert body["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"
    assert "requestId" in body


def test_oversized_byte_payload_raises_payload_too_large() -> None:
    from app.application.services.image_validation_service import ImageValidator
    from app.domain.errors import PayloadTooLargeError

    validator = ImageValidator(max_image_bytes=100)
    data = b"\xff\xd8" + b"x" * 200
    with pytest.raises(PayloadTooLargeError) as exc_info:
        validator.validate(data)
    assert "maximum size" in str(exc_info.value)


def test_oversized_dimensions_returns_422() -> None:
    data = _make_jpeg_header(60000, 60000)
    with _client() as client:
        response = client.post(
            "/api/v1/faces/recognize",
            files={"image": ("huge.jpg", BytesIO(data), "image/jpeg")},
        )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "INVALID_MEDIA"
    assert "requestId" in body


def test_valid_small_jpeg_passes_validation() -> None:
    data = _make_jpeg_header(16, 16)
    with _client() as client:
        response = client.post(
            "/api/v1/faces/recognize",
            files={"image": ("tiny.jpg", BytesIO(data), "image/jpeg")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["faceCount"] == 0
    assert "requestId" in body
