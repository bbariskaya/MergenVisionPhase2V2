"""Tests for the cross-store persistence orchestrator."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from mv_phase1_bulk.identities import EnrolledSample, SubjectBundle
from mv_phase1_bulk.persistence import PersistenceOrchestrator
from mv_phase1_bulk.types import FaceRecord, SampleRecord


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
    monkeypatch.setenv("MV_MINIO_ENDPOINT", "localhost:9000")
    monkeypatch.setenv("MV_MINIO_ACCESS_KEY", "test")
    monkeypatch.setenv("MV_MINIO_SECRET_KEY", "test")
    monkeypatch.setenv("MV_MINIO_BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("MV_QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("MV_PHASE1_BULK_ID_HMAC_KEY", "test-key")


def _sample_bundle(*, count: int = 2) -> SubjectBundle:
    face = FaceRecord(face_id="f1", display_name="Alice")
    samples: list[EnrolledSample] = []
    for i in range(count):
        sample = SampleRecord(sample_id=f"s{i}", face_id="f1", object_key=f"faces/f1/s{i}/aligned.jpg")
        samples.append(
            EnrolledSample(
                sample_record=sample,
                image_sha256=f"sha{i}",
                image_bytes=b"jpeg-bytes",
                crop_bytes=b"jpeg-bytes",
                embedding=np.ones(512, dtype=np.float32),
            )
        )
    return SubjectBundle(face=face, samples=samples)


def _upload_result(sample_id: str) -> MagicMock:
    result = MagicMock()
    result.object_key = f"faces/f1/{sample_id}/aligned.jpg"
    result.sha256 = "sha"
    result.bytes_written = 10
    return result


@pytest.fixture
def orchestrator() -> PersistenceOrchestrator:
    pg = MagicMock()
    pg.prepare_enrollment = AsyncMock()
    pg.activate_samples_tx = AsyncMock()
    pg.fail_samples_tx = AsyncMock()
    pg.deactivate_face_if_no_active_samples_tx = AsyncMock()

    minio = MagicMock()
    minio._bucket_name = "test-bucket"
    minio.upload_many = AsyncMock()
    minio.delete_best_effort = AsyncMock()

    qdrant = MagicMock()
    qdrant.upsert_many = AsyncMock()
    qdrant.delete_best_effort = AsyncMock()

    return PersistenceOrchestrator(
        postgres=pg,
        minio=minio,
        qdrant=qdrant,
        model_version="mv1",
    )


async def test_happy_path_persists_all_samples(orchestrator: PersistenceOrchestrator) -> None:
    bundle = _sample_bundle(count=2)
    orchestrator._minio.upload_many.return_value = [
        _upload_result("s0"),
        _upload_result("s1"),
    ]

    result = await orchestrator.persist_bundle(bundle)

    assert len(result.persisted) == 2
    assert len(result.failed) == 0
    orchestrator._postgres.prepare_enrollment.assert_awaited_once()
    orchestrator._minio.upload_many.assert_awaited_once()
    orchestrator._qdrant.upsert_many.assert_awaited_once()
    orchestrator._postgres.activate_samples_tx.assert_awaited_once()
    orchestrator._postgres.fail_samples_tx.assert_not_awaited()


async def test_minio_upload_failure_marks_sample_failed(
    orchestrator: PersistenceOrchestrator,
) -> None:
    bundle = _sample_bundle(count=2)
    exc = RuntimeError("network")
    orchestrator._minio.upload_many.return_value = [_upload_result("s0"), exc]

    result = await orchestrator.persist_bundle(bundle)

    assert len(result.persisted) == 1
    assert len(result.failed) == 1
    assert result.failed[0][0] == "s1"
    orchestrator._qdrant.upsert_many.assert_awaited_once()
    activations = orchestrator._postgres.activate_samples_tx.call_args[0][0]
    assert len(activations) == 1
    assert activations[0][0] == "s0"


async def test_qdrant_failure_rolls_back_minio(
    orchestrator: PersistenceOrchestrator,
) -> None:
    bundle = _sample_bundle(count=2)
    orchestrator._minio.upload_many.return_value = [
        _upload_result("s0"),
        _upload_result("s1"),
    ]
    orchestrator._qdrant.upsert_many.side_effect = RuntimeError("qdrant down")

    result = await orchestrator.persist_bundle(bundle)

    assert len(result.persisted) == 0
    assert len(result.failed) == 2
    assert orchestrator._minio.delete_best_effort.await_count == 2
    orchestrator._postgres.activate_samples_tx.assert_not_awaited()


async def test_pg_activation_failure_rolls_back_minio_and_qdrant(
    orchestrator: PersistenceOrchestrator,
) -> None:
    bundle = _sample_bundle(count=2)
    orchestrator._minio.upload_many.return_value = [
        _upload_result("s0"),
        _upload_result("s1"),
    ]
    orchestrator._postgres.activate_samples_tx.side_effect = RuntimeError("pg down")

    result = await orchestrator.persist_bundle(bundle)

    assert len(result.persisted) == 0
    assert len(result.failed) == 2
    assert orchestrator._minio.delete_best_effort.await_count == 2
    assert orchestrator._qdrant.delete_best_effort.await_count == 2


async def test_rejected_samples_reported_once_in_failed(orchestrator: PersistenceOrchestrator) -> None:
    bundle = _sample_bundle(count=1)
    orchestrator._minio.upload_many.return_value = [_upload_result("s0")]
    rejected = [("pre-existing", "no_face")]

    result = await orchestrator.persist_bundle(bundle, rejected=rejected)

    assert len(result.persisted) == 1
    assert result.failed == rejected
    orchestrator._postgres.fail_samples_tx.assert_awaited_once_with([("pre-existing", "no_face")])


async def test_no_accepted_samples_does_not_create_face_identity(
    orchestrator: PersistenceOrchestrator,
) -> None:
    bundle = _sample_bundle(count=1)
    bundle.samples = []
    rejected = [("s0", "no_face")]

    result = await orchestrator.persist_bundle(bundle, rejected=rejected)

    orchestrator._postgres.prepare_enrollment.assert_not_awaited()
    orchestrator._postgres.fail_samples_tx.assert_awaited_once_with(rejected)
    orchestrator._postgres.activate_samples_tx.assert_not_awaited()
    orchestrator._postgres.deactivate_face_if_no_active_samples_tx.assert_not_awaited()
    assert result.persisted == []
    assert result.failed == rejected


async def test_all_samples_failed_deactivates_empty_face_identity(
    orchestrator: PersistenceOrchestrator,
) -> None:
    bundle = _sample_bundle(count=2)
    orchestrator._minio.upload_many.return_value = [
        RuntimeError("network"),
        RuntimeError("network"),
    ]

    result = await orchestrator.persist_bundle(bundle)

    assert len(result.persisted) == 0
    assert len(result.failed) == 2
    orchestrator._postgres.deactivate_face_if_no_active_samples_tx.assert_awaited_once_with(bundle.face.face_id)
