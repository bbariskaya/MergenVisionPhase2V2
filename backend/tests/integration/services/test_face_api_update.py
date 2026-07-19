"""Integration tests for PATCH /api/v1/faces/{faceId} known identity update."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.controllers.face_controller import FaceController
from app.api.routes.dependencies import get_face_controller
from app.api.main import create_app
from app.application.ports.image_recognition import (
    ImageRecognitionEngine,
    NativeRecognitionResult,
)
from app.application.services.image_recognition_service import ImageRecognitionService
from app.domain.entities.face_identity import FaceIdentity
from app.domain.value_objects import FaceId
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork


class _NoOpEngine(ImageRecognitionEngine):
    async def detect_and_embed(self, image_bytes: bytes) -> NativeRecognitionResult:
        raise NotImplementedError("engine should not be called in update tests")


class _FakeLifecycle:
    pass


def _face_controller() -> FaceController:
    uow_factory = lambda: SqlAlchemyUnitOfWork(async_session_maker)  # noqa: E731
    service = ImageRecognitionService(
        lifecycle_service=_FakeLifecycle(),  # type: ignore[arg-type]
        unit_of_work_factory=uow_factory,
        max_image_bytes=10_000_000,
        model_version="v1",
        engine=_NoOpEngine(),
        match_threshold=0.55,
    )
    return FaceController(service, object_store=None)


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    app.dependency_overrides[get_face_controller] = _face_controller
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
async def known_identity() -> FaceIdentity:
    face_id = FaceId(uuid4())
    identity = FaceIdentity(
        face_id=face_id,
        status="known",
        is_active=True,
        display_name="Alice",
        identity_metadata={"department": "IT"},
        version=2,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    uow = SqlAlchemyUnitOfWork(async_session_maker)
    async with uow:
        await uow.face_identities.add(identity)
        await uow.commit()
    return identity


async def test_patch_update_changes_name_and_metadata(
    client: TestClient,
    known_identity: FaceIdentity,
) -> None:
    response = client.patch(
        f"/api/v1/faces/{known_identity.face_id}",
        json={"name": "Alice Smith", "metadata": {"department": "HR"}},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["faceId"] == str(known_identity.face_id)
    assert payload["status"] == "known"
    assert payload["name"] == "Alice Smith"
    assert payload["metadata"] == {"department": "HR"}

    uow = SqlAlchemyUnitOfWork(async_session_maker)
    async with uow:
        updated = await uow.face_identities.get_by_id(known_identity.face_id)
        assert updated is not None
        assert updated.display_name == "Alice Smith"
        assert updated.identity_metadata == {"department": "HR"}
        assert updated.version == 3


async def test_patch_update_preserves_face_id(
    client: TestClient,
    known_identity: FaceIdentity,
) -> None:
    response = client.patch(
        f"/api/v1/faces/{known_identity.face_id}",
        json={"name": "Renamed Alice"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["faceId"] == str(known_identity.face_id)


async def test_patch_update_inactive_identity_returns_error(
    client: TestClient,
) -> None:
    face_id = FaceId(uuid4())
    identity = FaceIdentity(
        face_id=face_id,
        status="known",
        is_active=True,
        display_name="Bob",
        identity_metadata={},
        version=1,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    identity.deactivate()
    uow = SqlAlchemyUnitOfWork(async_session_maker)
    async with uow:
        await uow.face_identities.add(identity)
        await uow.commit()

    response = client.patch(
        f"/api/v1/faces/{face_id}",
        json={"name": "Active Bob"},
    )
    assert response.status_code == 400, response.text
    assert "not found" in response.json()["error"]["message"].lower()


async def test_patch_update_nonexistent_identity_returns_error(
    client: TestClient,
) -> None:
    response = client.patch(
        f"/api/v1/faces/{uuid4()}",
        json={"name": "Ghost"},
    )
    assert response.status_code == 400, response.text
    assert "not found" in response.json()["error"]["message"].lower()


async def test_patch_update_anonymous_identity_returns_error(
    client: TestClient,
) -> None:
    face_id = FaceId(uuid4())
    identity = FaceIdentity(
        face_id=face_id,
        status="anonymous",
        is_active=True,
        display_name=None,
        identity_metadata={},
        version=1,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    uow = SqlAlchemyUnitOfWork(async_session_maker)
    async with uow:
        await uow.face_identities.add(identity)
        await uow.commit()

    response = client.patch(
        f"/api/v1/faces/{face_id}",
        json={"name": "Named"},
    )
    assert response.status_code == 400, response.text
    assert "Cannot update identity with status anonymous" in response.json()["error"]["message"]


