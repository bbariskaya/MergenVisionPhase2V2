"""Candidate selection validation integration tests.

These tests use a scripted VectorStore to inject exact candidate scores and
verify that the service:
- skips stale, malformed, inactive and non-finite candidates,
- falls through to the next valid candidate,
- clamps confidence to [0, 1],
- never leaves a process in ``processing`` after a failure.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from app.application.ports.id_generator import IdGenerator
from app.application.ports.object_store import ObjectStore
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.ports.vector_store import VectorCandidate, VectorStore
from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.domain.errors import IdentityResolutionError
from app.domain.value_objects import BoundingBox, FaceId, SampleId
from tests.fixtures.embedding_fixtures import vector_a, vector_b

BBOX = BoundingBox(x=0, y=0, width=16, height=16)
MATCH_THRESHOLD = 0.95


class ScriptedVectorStore(VectorStore):
    """Vector-store fake whose ``query`` returns pre-defined candidates."""

    def __init__(self, candidates: list[VectorCandidate] | None = None) -> None:
        self._candidates = candidates or []
        self.upsert_calls: list[tuple[SampleId, FaceId, Sequence[float]]] = []

    async def upsert(
        self,
        sample_id: SampleId,
        face_id: FaceId,
        embedding: Sequence[float],
    ) -> None:
        self.upsert_calls.append((sample_id, face_id, embedding))

    async def query(
        self,
        embedding: Sequence[float],
        top_k: int,
    ) -> Sequence[VectorCandidate]:
        return list(self._candidates)

    async def set_active(
        self,
        sample_id: SampleId,
        active: bool,
    ) -> None:
        pass

    async def delete(self, sample_id: SampleId) -> None:
        pass


class FailingVectorStore(VectorStore):
    """Vector-store fake that always raises on query."""

    async def upsert(
        self,
        sample_id: SampleId,
        face_id: FaceId,
        embedding: Sequence[float],
    ) -> None:
        pass

    async def query(
        self,
        embedding: Sequence[float],
        top_k: int,
    ) -> Sequence[VectorCandidate]:
        raise RuntimeError("simulated vector store outage")

    async def set_active(
        self,
        sample_id: SampleId,
        active: bool,
    ) -> None:
        pass

    async def delete(self, sample_id: SampleId) -> None:
        pass


@pytest.fixture
def scripted_service(
    unit_of_work_factory: UnitOfWorkFactory,
    object_store: ObjectStore,
    id_generator: IdGenerator,
) -> IdentityStorageLifecycleService:
    return IdentityStorageLifecycleService(
        unit_of_work_factory=unit_of_work_factory,
        object_store=object_store,
        vector_store=ScriptedVectorStore(),
        id_generator=id_generator,
    )


async def test_first_stale_candidate_skipped_to_valid_second(
    scripted_service: IdentityStorageLifecycleService,
    crop_bytes: bytes,
) -> None:
    # Seed an active identity and a deactivated identity, each with one sample.
    active = await scripted_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )
    assert active.sample_id is not None
    stale = await scripted_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_b(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )
    assert stale.sample_id is not None
    await scripted_service.deactivate_identity(stale.face_id)

    # Inject stale candidate first, valid candidate second.
    scripted_service._vector_store = ScriptedVectorStore(
        candidates=[
            VectorCandidate(
                sample_id=stale.sample_id,
                face_id=stale.face_id,
                score=0.99,
            ),
            VectorCandidate(
                sample_id=active.sample_id,
                face_id=active.face_id,
                score=0.98,
            ),
        ]
    )

    outcome = await scripted_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    assert outcome.face_id == active.face_id
    assert outcome.status == "anonymous"
    assert math.isfinite(outcome.match_confidence)


async def test_all_stale_candidates_result_in_new_anonymous(
    scripted_service: IdentityStorageLifecycleService,
    crop_bytes: bytes,
) -> None:
    stale = await scripted_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )
    assert stale.sample_id is not None
    await scripted_service.deactivate_identity(stale.face_id)

    scripted_service._vector_store = ScriptedVectorStore(
        candidates=[
            VectorCandidate(
                sample_id=stale.sample_id,
                face_id=stale.face_id,
                score=0.99,
            ),
        ]
    )

    outcome = await scripted_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_b(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    assert outcome.status == "new_anonymous"
    assert outcome.face_id != stale.face_id
    # Stale high score must not elevate confidence for a new identity.
    assert outcome.match_confidence == 0.0


async def test_malformed_candidate_skipped(
    scripted_service: IdentityStorageLifecycleService,
    crop_bytes: bytes,
    id_generator: IdGenerator,
) -> None:
    active = await scripted_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )
    assert active.sample_id is not None

    fake_sample_id = SampleId(id_generator.new_uuid7())
    scripted_service._vector_store = ScriptedVectorStore(
        candidates=[
            VectorCandidate(
                sample_id=fake_sample_id,
                face_id=active.face_id,
                score=0.99,
            ),
        ]
    )

    outcome = await scripted_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_b(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    assert outcome.status == "new_anonymous"


async def test_inactive_sample_candidate_skipped(
    scripted_service: IdentityStorageLifecycleService,
    crop_bytes: bytes,
) -> None:
    first = await scripted_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )
    assert first.sample_id is not None
    # Add a second sample and mark it inactive via deactivation/reactivation workaround.
    second_sample = await scripted_service.add_sample(
        face_id=first.face_id,
        crop_bytes=crop_bytes,
        embedding=vector_b(),
    )
    assert second_sample.sample_id is not None
    # Manually inactivate the second sample through the repository.
    from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
    from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork

    async with SqlAlchemyUnitOfWork(async_session_maker) as uow:
        loaded = await uow.face_samples.get_by_id(second_sample.sample_id)
        assert loaded is not None
        loaded.mark_inactive()
        await uow.face_samples.update(loaded)
        await uow.commit()

    scripted_service._vector_store = ScriptedVectorStore(
        candidates=[
            VectorCandidate(
                sample_id=second_sample.sample_id,
                face_id=first.face_id,
                score=0.99,
            ),
            VectorCandidate(
                sample_id=first.sample_id,
                face_id=first.face_id,
                score=0.98,
            ),
        ]
    )

    outcome = await scripted_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    assert outcome.status == "anonymous"
    assert outcome.sample_id == first.sample_id


async def test_exact_threshold_boundary_accepted(
    scripted_service: IdentityStorageLifecycleService,
    crop_bytes: bytes,
) -> None:
    first = await scripted_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )
    assert first.sample_id is not None

    scripted_service._vector_store = ScriptedVectorStore(
        candidates=[
            VectorCandidate(
                sample_id=first.sample_id,
                face_id=first.face_id,
                score=MATCH_THRESHOLD,
            ),
        ]
    )

    outcome = await scripted_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    assert outcome.face_id == first.face_id
    assert outcome.status == "anonymous"


async def test_non_finite_candidates_skipped(
    scripted_service: IdentityStorageLifecycleService,
    crop_bytes: bytes,
    id_generator: IdGenerator,
) -> None:
    face_id = FaceId(id_generator.new_uuid7())
    sample_id = SampleId(id_generator.new_uuid7())
    scripted_service._vector_store = ScriptedVectorStore(
        candidates=[
            VectorCandidate(sample_id=sample_id, face_id=face_id, score=float("nan")),
            VectorCandidate(sample_id=sample_id, face_id=face_id, score=float("inf")),
            VectorCandidate(sample_id=sample_id, face_id=face_id, score=float("-inf")),
        ]
    )

    outcome = await scripted_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    assert outcome.status == "new_anonymous"
    assert outcome.match_confidence == 0.0


async def test_vector_query_failure_fails_process_and_creates_nothing(
    unit_of_work_factory: UnitOfWorkFactory,
    object_store: ObjectStore,
    id_generator: IdGenerator,
    crop_bytes: bytes,
) -> None:
    import os

    import asyncpg

    service = IdentityStorageLifecycleService(
        unit_of_work_factory=unit_of_work_factory,
        object_store=object_store,
        vector_store=FailingVectorStore(),
        id_generator=id_generator,
    )

    with pytest.raises(IdentityResolutionError):
        await service.resolve_or_create(
            crop_bytes=crop_bytes,
            embedding=vector_a(),
            bbox=BBOX,
            match_threshold=MATCH_THRESHOLD,
        )

    url = os.environ["DATABASE_URL"].replace("postgresql+asyncpg", "postgresql")
    conn = await asyncpg.connect(url)
    try:
        identity_count = await conn.fetchval("SELECT count(*) FROM face_identity")
        sample_count = await conn.fetchval("SELECT count(*) FROM face_sample")
        result_count = await conn.fetchval("SELECT count(*) FROM recognition_result")
        process_row = await conn.fetchrow("SELECT status, error_code FROM process_record LIMIT 1")
    finally:
        await conn.close()

    assert identity_count == 0
    assert sample_count == 0
    assert result_count == 0
    assert process_row is not None
    assert process_row["status"] == "failed"
    assert process_row["error_code"] == "vector_query_failed"


async def test_negative_cosine_yields_new_anonymous_with_zero_confidence(
    lifecycle_service: IdentityStorageLifecycleService,
    crop_bytes: bytes,
) -> None:
    first = await lifecycle_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=vector_a(),
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )
    assert first.sample_id is not None

    negative_a = [-x for x in vector_a()]
    outcome = await lifecycle_service.resolve_or_create(
        crop_bytes=crop_bytes,
        embedding=negative_a,
        bbox=BBOX,
        match_threshold=MATCH_THRESHOLD,
    )

    assert outcome.status == "new_anonymous"
    assert outcome.face_id != first.face_id
    assert outcome.match_confidence == 0.0
