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
from app.domain.value_objects import BoundingBox, FaceId, PersonId, ProcessId, SampleId


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
                self.person_id = None
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
                self.person_id = None
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

    async def list_identities(self, query: str | None = None, status: str | None = None) -> list[Any]:
        items: list[Any] = []
        for face_id, (name, metadata) in self.known_identities.items():
            if status and status != "known":
                continue
            if query and query.lower() not in name.lower():
                continue

            class _Identity:
                def __init__(self, face_id: FaceId, name: str, metadata: dict[str, Any]) -> None:
                    self.face_id = face_id
                    self.status = "known"
                    self.display_name = name
                    self.identity_metadata = metadata
                    self.created_at = None
                    self.updated_at = None

            items.append(_Identity(face_id, name, metadata))
        return items

    async def list_face_samples(self, face_id: FaceId) -> list[Any]:
        class _Sample:
            def __init__(self, sample_id: str, face_id: FaceId) -> None:
                self.sample_id = sample_id
                self.face_id = face_id
                self.state = "active"
                self.object_key = None
                self.created_at = None
                self.activated_at = None
        return [_Sample("018f0000-0000-0000-0000-0000000000a1", face_id)]

    async def add_face_sample(self, face_id: FaceId, image_bytes: bytes) -> Any:
        class _Sample:
            def __init__(self, sample_id: str, face_id: FaceId) -> None:
                self.sample_id = SampleId(UUID(sample_id))
                self.face_id = face_id
                self.state = "active"
                self.object_key = f"faces/{face_id}/sample.jpg"
                self.created_at = None
                self.activated_at = None
        return _Sample("018f0000-0000-0000-0000-0000000000a2", face_id)

    async def delete_face_sample(self, face_id: FaceId, sample_id: SampleId) -> None:
        pass

    async def assign_face_to_person(
        self,
        face_id: FaceId,
        target_person_id: PersonId,
    ) -> Any:
        if face_id in self.deleted:
            raise ValueError("Face identity not found")
        name, metadata = self.known_identities.get(face_id, (None, {}))
        class _Identity:
            def __init__(self, face_id: FaceId, person_id: PersonId) -> None:
                self.face_id = face_id
                self.status = "known"
                self.display_name = name or ""
                self.identity_metadata = metadata
                self.person_id = person_id
                self.redirect_to_face_id = None
        return _Identity(face_id, target_person_id)


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
    response = client.post("/api/v1/faces/recognize", files={"image": ("no_face.jpg", BytesIO(_valid_jpeg_header()), "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["faceCount"] == 0
    assert data["faces"] == []
    assert "processId" in data


def test_recognize_single_face_returns_new_anonymous(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=10, y=20, width=30, height=40), confidence=0.92),
    ]
    response = client.post("/api/v1/faces/recognize", files={"image": ("single.jpg", BytesIO(_valid_jpeg_header()), "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["faceCount"] == 1
    face = data["faces"][0]
    assert face["status"] == "new_anonymous"
    assert face["name"] is None
    assert face["metadata"] is None
    assert face["boundingBox"] == {"x": 10, "y": 20, "width": 30, "height": 40}


def test_recognize_same_face_twice_returns_anonymous_then_known(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=10, y=20, width=30, height=40), confidence=0.94),
    ]
    first = client.post("/api/v1/faces/recognize", files={"image": ("single.jpg", BytesIO(_valid_jpeg_header()), "image/jpeg")})
    face_id = first.json()["faces"][0]["faceId"]

    second = client.post("/api/v1/faces/recognize", files={"image": ("single2.jpg", BytesIO(_valid_jpeg_header()), "image/jpeg")})
    face = second.json()["faces"][0]
    assert face["faceId"] == face_id
    assert face["status"] == "anonymous"

    response = client.post(f"/api/v1/faces/{face_id}/enroll", json={"name": "Rachel", "metadata": {"role": "friend"}})
    assert response.status_code == 200
    assert response.json()["status"] == "known"

    third = client.post("/api/v1/faces/recognize", files={"image": ("single3.jpg", BytesIO(_valid_jpeg_header()), "image/jpeg")})
    face = third.json()["faces"][0]
    assert face["faceId"] == face_id
    assert face["status"] == "known"
    assert face["name"] == "Rachel"
    assert face["metadata"] == {"role": "friend"}


def test_recognize_multi_faces_returns_separate_results(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=10, y=20, width=30, height=40), confidence=0.92),
        _FakeDetection(bbox=BoundingBox(x=100, y=200, width=50, height=50), confidence=0.87),
    ]
    response = client.post("/api/v1/faces/recognize", files={"image": ("multi.jpg", BytesIO(_valid_jpeg_header()), "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["faceCount"] == 2
    assert len({f["faceId"] for f in data["faces"]}) == 2
    assert all(f["status"] == "new_anonymous" for f in data["faces"])


def test_enroll_existing_anonymous_returns_known(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=10, y=20, width=30, height=40), confidence=0.9),
    ]
    recognized = client.post("/api/v1/faces/recognize", files={"image": ("x.jpg", BytesIO(_valid_jpeg_header()), "image/jpeg")})
    face_id = recognized.json()["faces"][0]["faceId"]

    response = client.post(f"/api/v1/faces/{face_id}/enroll", json={"name": "Rachel"})
    assert response.status_code == 200
    data = response.json()
    assert data["faceId"] == face_id
    assert data["status"] == "known"
    assert data["name"] == "Rachel"


def test_get_face_detail_returns_known_identity(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=1, y=2, width=3, height=4), confidence=0.9),
    ]
    recognized = client.post("/api/v1/faces/recognize", files={"image": ("x.jpg", BytesIO(_valid_jpeg_header()), "image/jpeg")})
    face_id = recognized.json()["faces"][0]["faceId"]
    client.post(f"/api/v1/faces/{face_id}/enroll", json={"name": "Rachel", "metadata": {"show": "Friends"}})

    response = client.get(f"/api/v1/faces/{face_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["faceId"] == face_id
    assert data["status"] == "known"
    assert data["name"] == "Rachel"
    assert data["metadata"] == {"show": "Friends"}


def test_delete_face_returns_ok_and_blocks_detail(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=1, y=2, width=3, height=4), confidence=0.9),
    ]
    recognized = client.post("/api/v1/faces/recognize", files={"image": ("x.jpg", BytesIO(_valid_jpeg_header()), "image/jpeg")})
    face_id = recognized.json()["faces"][0]["faceId"]

    response = client.delete(f"/api/v1/faces/{face_id}")
    assert response.status_code == 200
    assert "requestId" in response.json()

    detail = client.get(f"/api/v1/faces/{face_id}")
    assert detail.status_code == 404
    assert detail.json()["error"]["code"] == "FACE_NOT_FOUND"


def test_get_face_history_lists_processes(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=1, y=2, width=3, height=4), confidence=0.9),
    ]
    recognized = client.post("/api/v1/faces/recognize", files={"image": ("x.jpg", BytesIO(_valid_jpeg_header()), "image/jpeg")})
    face_id = recognized.json()["faces"][0]["faceId"]

    response = client.get(f"/api/v1/faces/{face_id}/history")
    assert response.status_code == 200
    data = response.json()
    assert data["faceId"] == face_id
    assert len(data["history"]) >= 1
    assert "processId" in data["history"][0]


def test_get_process_returns_completed_details(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=1, y=2, width=3, height=4), confidence=0.9),
    ]
    recognized = client.post("/api/v1/faces/recognize", files={"image": ("x.jpg", BytesIO(_valid_jpeg_header()), "image/jpeg")})
    process_id = recognized.json()["processId"]

    response = client.get(f"/api/v1/processes/{process_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["processId"] == process_id
    assert data["processType"] == "image_recognize"
    assert data["status"] == "completed"
    assert data["faceCount"] == 1


def test_list_identities_returns_known_faces(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=1, y=2, width=3, height=4), confidence=0.9),
    ]
    recognized = client.post("/api/v1/faces/recognize", files={"image": ("x.jpg", BytesIO(_valid_jpeg_header()), "image/jpeg")})
    face_id = recognized.json()["faces"][0]["faceId"]
    client.post(f"/api/v1/faces/{face_id}/enroll", json={"name": "Rachel"})

    response = client.get("/api/v1/faces")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert len(data["identities"]) == 1
    assert data["identities"][0]["faceId"] == face_id
    assert data["identities"][0]["status"] == "known"
    assert data["identities"][0]["name"] == "Rachel"


def test_list_identities_supports_search_query(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=1, y=2, width=3, height=4), confidence=0.9),
    ]
    recognized = client.post("/api/v1/faces/recognize", files={"image": ("x.jpg", BytesIO(_valid_jpeg_header()), "image/jpeg")})
    face_id = recognized.json()["faces"][0]["faceId"]
    client.post(f"/api/v1/faces/{face_id}/enroll", json={"name": "Rachel"})

    response = client.get("/api/v1/faces?search=monica")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["identities"] == []

    response = client.get("/api/v1/faces?search=rachel")
    data = response.json()
    assert data["count"] == 1


def test_list_face_samples_returns_samples(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=1, y=2, width=3, height=4), confidence=0.9),
    ]
    recognized = client.post("/api/v1/faces/recognize", files={"image": ("x.jpg", BytesIO(_valid_jpeg_header()), "image/jpeg")})
    face_id = recognized.json()["faces"][0]["faceId"]

    response = client.get(f"/api/v1/faces/{face_id}/samples")
    assert response.status_code == 200
    data = response.json()
    assert data["faceId"] == face_id
    assert data["count"] == 1
    assert "sampleId" in data["samples"][0]
    assert data["samples"][0]["state"] == "active"


def test_add_face_sample_returns_new_sample(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=1, y=2, width=3, height=4), confidence=0.9),
    ]
    recognized = client.post("/api/v1/faces/recognize", files={"image": ("x.jpg", BytesIO(_valid_jpeg_header()), "image/jpeg")})
    face_id = recognized.json()["faces"][0]["faceId"]

    response = client.post(
        f"/api/v1/faces/{face_id}/samples",
        files={"image": ("add.jpg", BytesIO(_valid_jpeg_header()), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["faceId"] == face_id
    assert data["state"] == "active"
    assert "/image" in data["imageUrl"]


def test_assign_face_to_existing_person_returns_known(client: TestClient, fake_controller: FaceController) -> None:
    fake_controller._service.next_detections = [
        _FakeDetection(bbox=BoundingBox(x=1, y=2, width=3, height=4), confidence=0.9),
    ]
    recognized = client.post("/api/v1/faces/recognize", files={"image": ("x.jpg", BytesIO(_valid_jpeg_header()), "image/jpeg")})
    face_id = recognized.json()["faces"][0]["faceId"]
    person_id = "018f0000-0000-7000-8000-000000000001"

    response = client.post(
        f"/api/v1/faces/{face_id}/assign",
        json={"personId": person_id},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["faceId"] == face_id
    assert data["status"] == "known"
    assert data["personId"] == person_id
    assert "requestId" in data
