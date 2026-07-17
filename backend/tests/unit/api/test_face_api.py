"""Unit tests for the face recognition HTTP API.

These tests exercise the FastAPI routes with the controller injected,
verifying request/response contracts and business semantics.
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


@dataclass(frozen=True)
class _FakeDetection:
    bbox: BoundingBox
    confidence: float


class _FakeImageRecognitionService:
    def __init__(self) -> None:
        self.next_detections: list[_FakeDetection] = []
        self.resolve_counter: dict[FaceId, int] = {}
        self.known_identities: dict[FaceId, tuple[str, dict[str, Any]]] = {}
        self.deleted: set[FaceId] = set()

    async def recognize_image(self, image_bytes: bytes) -> RecognizeResult:
        process = ProcessRecord(
            process_id=ProcessId(UUID("018f0000-0000-7b0e-8000-000000000001")),
            process_type="image_recognize",
            status="completed",
            face_count=len(self.next_detections),
        )
        faces: list[RecognitionResultItem] = []
        for idx, det in enumerate(self.next_detections):
            face_id = FaceId(UUID(f"018f0000-0000-7b0e-8000-0000000000{idx + 2:02d}"))
            if face_id in self.deleted:
                continue
            count = self.resolve_counter.get(face_id, 0)
            self.resolve_counter[face_id] = count + 1
            if face_id in self.known_identities:
                status = "known"
                name, metadata = self.known_identities[face_id]
            elif count == 0:
                status = "new_anonymous"
                name = None
                metadata = {}
            else:
                status = "anonymous"
                name = None
                metadata = {}
            faces.append(
                RecognitionResultItem(
                    face_id=face_id,
                    status=status,
                    name=name,
                    metadata=metadata,
                    bounding_box=det.bbox,
                    confidence=det.confidence,
                )
            )
        if not faces:
            process.face_count = 0
        return RecognizeResult(process=process, faces=faces)

    async def enroll_face(
        self, face_id: FaceId, display_name: str, metadata: dict[str, Any]
    ) -> Any:
        if face_id in self.deleted:
            raise ValueError("Face identity not found")
        self.known_identities[face_id] = (display_name, metadata)
        class _Identity:
            def __init__(self, face_id: FaceId, display_name: str, metadata: dict[str, Any]) -> None:
                self.face_id = face_id
                self.status = "known"
                self.display_name = display_name
                self.identity_metadata = metadata
        return _Identity(face_id, display_name, metadata)

    async def get_identity_detail(self, face_id: FaceId) -> Any:
        if face_id in self.deleted:
            return None
        name, metadata = self.known_identities.get(face_id, (None, {}))
        status = "known" if name else "anonymous"
        class _Identity:
            def __init__(self, face_id: FaceId, status: str, name: str | None, metadata: dict[str, Any]) -> None:
                self.face_id = face_id
                self.status = status
                self.display_name = name
                self.identity_metadata = metadata
                self.created_at = None
                self.updated_at = None
        return _Identity(face_id, status, name, metadata)

    async def delete_identity(self, face_id: FaceId) -> bool:
        if face_id in self.deleted:
            raise ValueError("Face identity not found")
        self.deleted.add(face_id)
        return True

    async def get_face_history(self, face_id: FaceId, limit: int = 100) -> list[dict[str, Any]]:
        if face_id in self.deleted:
            return []
        return [
            {
                "process_id": "018f0000-0000-7b0e-8000-000000000001",
                "timestamp": "2026-07-17T00:00:00+00:00",
                "process_type": "image_recognize",
                "status": "completed",
                "recognition_status": "new_anonymous",
                "match_confidence": 0.9,
            }
        ]

    async def get_process(self, process_id: ProcessId) -> Any:
        class _Process:
            def __init__(self, process_id: ProcessId) -> None:
                self.process_id = process_id
                self.process_type = "image_recognize"
                self.status = "completed"
                self.face_count = 1
                self.error_code = None
                self.details = {}
                self.created_at = None
                self.completed_at = None
        return _Process(process_id)


@pytest.fixture
def fake_controller() -> FaceController:
    return FaceController(_FakeImageRecognitionService())


@pytest.fixture
def client(fake_controller: FaceController) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_face_controller] = lambda: fake_controller
    app.state.face_controller = fake_controller
    return TestClient(app)


def test_recognize_no_face_returns_completed_with_zero_faces(client: TestClient) -> None:
    response = client.post("/faces/recognize", files={"image": ("no_face.jpg", BytesIO(b"fake-jpeg"), "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["face_count"] == 0
    assert data["faces"] == []
    assert "process_id" in data


def test_recognize_single_face_returns_new_anonymous(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=10, y=20, width=30, height=40), confidence=0.92),
    ]
    response = client.post("/faces/recognize", files={"image": ("single.jpg", BytesIO(b"fake-jpeg"), "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["face_count"] == 1
    face = data["faces"][0]
    assert face["status"] == "new_anonymous"
    assert face["name"] is None
    assert face["metadata"] is None
    assert face["bounding_box"] == {"x": 10, "y": 20, "width": 30, "height": 40}


def test_recognize_same_face_twice_returns_anonymous_then_known(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=10, y=20, width=30, height=40), confidence=0.94),
    ]
    first = client.post("/faces/recognize", files={"image": ("single.jpg", BytesIO(b"fake-jpeg"), "image/jpeg")})
    face_id = first.json()["faces"][0]["face_id"]

    second = client.post("/faces/recognize", files={"image": ("single2.jpg", BytesIO(b"fake-jpeg"), "image/jpeg")})
    face = second.json()["faces"][0]
    assert face["face_id"] == face_id
    assert face["status"] == "anonymous"

    response = client.post("/faces/enroll", json={"face_id": face_id, "name": "Rachel", "metadata": {"role": "friend"}})
    assert response.status_code == 200
    assert response.json()["status"] == "known"

    third = client.post("/faces/recognize", files={"image": ("single3.jpg", BytesIO(b"fake-jpeg"), "image/jpeg")})
    face = third.json()["faces"][0]
    assert face["face_id"] == face_id
    assert face["status"] == "known"
    assert face["name"] == "Rachel"
    assert face["metadata"] == {"role": "friend"}


def test_recognize_multi_faces_returns_separate_results(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=10, y=20, width=30, height=40), confidence=0.92),
        _FakeDetection(bbox=BoundingBox(x=100, y=200, width=50, height=50), confidence=0.87),
    ]
    response = client.post("/faces/recognize", files={"image": ("multi.jpg", BytesIO(b"fake-jpeg"), "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["face_count"] == 2
    assert len({f["face_id"] for f in data["faces"]}) == 2
    assert all(f["status"] == "new_anonymous" for f in data["faces"])


def test_enroll_existing_anonymous_returns_known(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=10, y=20, width=30, height=40), confidence=0.9),
    ]
    recognized = client.post("/faces/recognize", files={"image": ("x.jpg", BytesIO(b"fake-jpeg"), "image/jpeg")})
    face_id = recognized.json()["faces"][0]["face_id"]

    response = client.post("/faces/enroll", json={"face_id": face_id, "name": "Rachel"})
    assert response.status_code == 200
    data = response.json()
    assert data["face_id"] == face_id
    assert data["status"] == "known"
    assert data["name"] == "Rachel"


def test_get_face_detail_returns_known_identity(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=1, y=2, width=3, height=4), confidence=0.9),
    ]
    recognized = client.post("/faces/recognize", files={"image": ("x.jpg", BytesIO(b"fake-jpeg"), "image/jpeg")})
    face_id = recognized.json()["faces"][0]["face_id"]
    client.post("/faces/enroll", json={"face_id": face_id, "name": "Rachel", "metadata": {"show": "Friends"}})

    response = client.get(f"/faces/{face_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["face_id"] == face_id
    assert data["status"] == "known"
    assert data["name"] == "Rachel"
    assert data["metadata"] == {"show": "Friends"}


def test_delete_face_returns_no_content(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=1, y=2, width=3, height=4), confidence=0.9),
    ]
    recognized = client.post("/faces/recognize", files={"image": ("x.jpg", BytesIO(b"fake-jpeg"), "image/jpeg")})
    face_id = recognized.json()["faces"][0]["face_id"]

    response = client.delete(f"/faces/{face_id}")
    assert response.status_code == 204

    detail = client.get(f"/faces/{face_id}")
    assert detail.status_code == 404


def test_get_face_history_lists_processes(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=1, y=2, width=3, height=4), confidence=0.9),
    ]
    recognized = client.post("/faces/recognize", files={"image": ("x.jpg", BytesIO(b"fake-jpeg"), "image/jpeg")})
    face_id = recognized.json()["faces"][0]["face_id"]

    response = client.get(f"/faces/{face_id}/history")
    assert response.status_code == 200
    data = response.json()
    assert data["face_id"] == face_id
    assert len(data["history"]) >= 1
    assert "process_id" in data["history"][0]


def test_get_process_returns_completed_details(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=1, y=2, width=3, height=4), confidence=0.9),
    ]
    recognized = client.post("/faces/recognize", files={"image": ("x.jpg", BytesIO(b"fake-jpeg"), "image/jpeg")})
    process_id = recognized.json()["process_id"]

    response = client.get(f"/processes/{process_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["process_id"] == process_id
    assert data["process_type"] == "image_recognize"
    assert data["status"] == "completed"
    assert data["face_count"] == 1
