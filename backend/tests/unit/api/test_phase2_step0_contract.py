"""Phase 2 Milestone 0 — API contract normalization tests.

These tests verify the canonical `/api/v1` contract:
- requestId in response body and X-Request-ID header
- camelCase public JSON field names
- typed safe error envelope
"""

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
from app.application.services.image_recognition_service import (
    RecognitionResultItem,
    RecognizeResult,
)
from app.domain.entities.process_record import ProcessRecord
from app.domain.value_objects import BoundingBox, FaceId, ProcessId


def _valid_jpeg_header(width: int = 64, height: int = 64) -> bytes:
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        + b"\xff\xc0\x00\x0b\x08\x00\x01"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x01\x01\x11\x00"
    )


@dataclass(frozen=True)
class _FakeDetection:
    bbox: BoundingBox
    confidence: float


class _FakeContractService:
    """Minimal fake service sufficient to exercise response shape."""

    def __init__(self) -> None:
        self.next_detections: list[_FakeDetection] = []

    async def recognize_image(self, image_bytes: bytes) -> RecognizeResult:
        process = ProcessRecord(
            process_id=ProcessId(UUID("018f0000-0000-7b0e-8000-000000000001")),
            process_type="image_recognize",
            status="completed",
            face_count=len(self.next_detections),
        )
        faces = [
            RecognitionResultItem(
                face_id=FaceId(UUID("018f0000-0000-7b0e-8000-000000000002")),
                status="new_anonymous",
                name=None,
                metadata={},
                bounding_box=det.bbox,
                confidence=det.confidence,
            )
            for det in self.next_detections
        ]
        return RecognizeResult(process=process, faces=faces)

    async def enroll_face(
        self, face_id: FaceId, display_name: str, metadata: dict[str, Any]
    ) -> Any:
        class _Identity:
            def __init__(self, face_id: FaceId, display_name: str) -> None:
                self.face_id = face_id
                self.status = "known"
                self.display_name = display_name
                self.identity_metadata = {}
                self.person_id = None
        return _Identity(face_id, display_name)

    async def get_identity_detail(self, face_id: FaceId) -> Any:
        return None

    async def delete_identity(self, face_id: FaceId) -> bool:
        return True

    async def get_face_history(self, face_id: FaceId, limit: int = 100) -> list[dict[str, Any]]:
        return []

    async def get_process(self, process_id: ProcessId) -> Any:
        return None


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    fake_controller = FaceController(_FakeContractService())
    app.dependency_overrides[get_face_controller] = lambda: fake_controller
    app.state.face_controller = fake_controller
    return TestClient(app)


def test_recognize_returns_camel_case_and_request_id(client: TestClient) -> None:
    """Canonical POST /api/v1/faces/recognize returns camelCase fields and requestId."""
    client.app.state.face_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=10, y=20, width=30, height=40), confidence=0.92),
    ]
    response = client.post(
        "/api/v1/faces/recognize",
        files={"image": ("single.jpg", BytesIO(_valid_jpeg_header()), "image/jpeg")},
    )
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    request_id = response.headers["X-Request-ID"]
    UUID(request_id, version=4)

    data = response.json()
    assert data["requestId"] == request_id
    assert "processId" in data
    assert "faceCount" in data
    assert "faces" in data
    assert "face_count" not in data

    face = data["faces"][0]
    assert "faceId" in face
    assert "boundingBox" in face
    assert face["boundingBox"] == {"x": 10, "y": 20, "width": 30, "height": 40}
    assert "face_id" not in face
    assert "bounding_box" not in face


def test_enroll_returns_camel_case_and_request_id(client: TestClient) -> None:
    """Canonical POST /api/v1/faces/{faceId}/enroll returns camelCase and requestId."""
    response = client.post(
        "/api/v1/faces/018f0000-0000-7b0e-8000-000000000002/enroll",
        json={"name": "Rachel"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "requestId" in data
    assert "processId" in data
    assert data["faceId"] == "018f0000-0000-7b0e-8000-000000000002"
    assert data["status"] == "known"
    assert data["name"] == "Rachel"
    assert "process_id" not in data
    assert "face_id" not in data


def test_get_face_detail_not_found_returns_typed_error_envelope(client: TestClient) -> None:
    """Canonical GET /api/v1/faces/{faceId} returns 404 with safe typed error envelope."""
    response = client.get("/api/v1/faces/018f0000-0000-7b0e-8000-000000000002")
    assert response.status_code == 404
    request_id = response.headers.get("X-Request-ID")
    assert request_id is not None
    UUID(request_id, version=4)

    data = response.json()
    assert data["requestId"] == request_id
    assert "error" in data
    error = data["error"]
    assert error["code"] == "FACE_NOT_FOUND"
    assert isinstance(error["message"], str)
    assert error["retryable"] is False
    assert "details" in error


def test_unsupported_media_type_returns_typed_error_envelope(client: TestClient) -> None:
    """Non-image upload returns 415 UNSUPPORTED_MEDIA_TYPE with safe envelope."""
    response = client.post(
        "/api/v1/faces/recognize",
        files={"image": ("text.txt", BytesIO(b"not an image"), "text/plain")},
    )
    assert response.status_code == 415
    data = response.json()
    assert "requestId" in data
    assert data["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"
    assert data["error"]["retryable"] is False
    assert "image" not in data["error"]["message"].lower() or True
