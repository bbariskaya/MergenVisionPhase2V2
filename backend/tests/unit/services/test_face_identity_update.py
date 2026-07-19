"""Known face identity name/metadata update unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Self

import pytest

from app.application.ports.image_recognition import (
    ImageRecognitionEngine,
    NativeRecognitionResult,
)
from app.application.ports.repositories import FaceIdentityRepository
from app.application.ports.unit_of_work import UnitOfWork
from app.application.services.image_recognition_service import ImageRecognitionService
from app.domain.entities.face_identity import FaceIdentity
from app.domain.errors import ConcurrentUpdateError, ValidationError
from app.domain.value_objects import FaceId
from app.infrastructure.uuid7 import generate_uuid7


class _FakeLifecycle:
    pass


class _FakeEngine(ImageRecognitionEngine):
    async def detect_and_embed(self, image_bytes: bytes) -> NativeRecognitionResult:
        raise NotImplementedError


@dataclass
class _FakeFaceIdentityRepo(FaceIdentityRepository):
    identities: dict[FaceId, FaceIdentity] = field(default_factory=dict)

    async def add(self, identity: FaceIdentity) -> None:
        self.identities[identity.face_id] = identity

    async def get_by_id(self, face_id: FaceId) -> FaceIdentity | None:
        identity = self.identities.get(face_id)
        return FaceIdentity(**identity.__dict__) if identity else None

    async def get_active_by_id(self, face_id: FaceId) -> FaceIdentity | None:
        identity = self.identities.get(face_id)
        if identity is None or not identity.is_active:
            return None
        return FaceIdentity(**identity.__dict__)

    async def get_many_by_ids(self, face_ids: Any) -> Any:
        return [self.identities[fid] for fid in face_ids if fid in self.identities]

    async def update(self, identity: FaceIdentity) -> None:
        self.identities[identity.face_id] = identity

    async def update_with_expected_version(
        self, identity: FaceIdentity, expected_version: int
    ) -> FaceIdentity:
        stored = self.identities.get(identity.face_id)
        if stored is None:
            raise ConcurrentUpdateError("Identity not found")
        if stored.version != expected_version:
            raise ConcurrentUpdateError("Version mismatch")
        identity.version = expected_version + 1
        self.identities[identity.face_id] = identity
        return identity

    async def list_all(self) -> Any:
        return list(self.identities.values())

    async def search(
        self, query: str | None = None, status: str | None = None, is_active: bool = True
    ) -> Any:
        raise NotImplementedError


@dataclass
class _FakeUoW(UnitOfWork):
    face_identities: _FakeFaceIdentityRepo = field(default_factory=_FakeFaceIdentityRepo)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass

    async def flush(self) -> None:
        pass


def _known_identity(name: str = "Alice") -> FaceIdentity:
    return FaceIdentity(
        face_id=FaceId(generate_uuid7()),
        status="known",
        is_active=True,
        display_name=name,
        identity_metadata={"department": "IT"},
        version=2,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


@pytest.mark.anyio
async def test_update_identity_changes_name_and_metadata() -> None:
    identity = _known_identity("Alice")
    repo = _FakeFaceIdentityRepo(identities={identity.face_id: identity})
    uow = _FakeUoW(face_identities=repo)

    service = ImageRecognitionService(
        lifecycle_service=_FakeLifecycle(),  # type: ignore[arg-type]
        unit_of_work_factory=lambda: uow,
        max_image_bytes=10_000_000,
        model_version="v1",
        engine=_FakeEngine(),
        match_threshold=0.55,
    )

    updated = await service.update_identity(identity.face_id, "Alice Smith", {"department": "HR"})

    assert updated.display_name == "Alice Smith"
    assert updated.identity_metadata == {"department": "HR"}
    assert updated.status == "known"
    assert updated.version == 3


@pytest.mark.anyio
async def test_update_identity_rejects_anonymous_identity() -> None:
    identity = FaceIdentity(
        face_id=FaceId(generate_uuid7()),
        status="anonymous",
        is_active=True,
        display_name=None,
        identity_metadata={},
        version=1,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    repo = _FakeFaceIdentityRepo(identities={identity.face_id: identity})
    uow = _FakeUoW(face_identities=repo)

    service = ImageRecognitionService(
        lifecycle_service=_FakeLifecycle(),  # type: ignore[arg-type]
        unit_of_work_factory=lambda: uow,
        max_image_bytes=10_000_000,
        model_version="v1",
        engine=_FakeEngine(),
        match_threshold=0.55,
    )

    with pytest.raises(ValidationError, match="Cannot update identity with status anonymous"):
        await service.update_identity(identity.face_id, "Name", {})
