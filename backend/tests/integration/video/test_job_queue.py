"""M3 integration tests: atomic job queue claim/lease/heartbeat/retry."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from app.application.ports.unit_of_work import UnitOfWork, UnitOfWorkFactory
from app.domain.entities.process_record import ProcessRecord
from app.domain.entities.video_asset import VideoAsset
from app.domain.entities.video_job import VideoJob
from app.domain.value_objects import JobId, ProcessId, UploadSessionId, VideoId
from app.infrastructure.uuid7 import generate_uuid7


@pytest.fixture(autouse=True)
async def clean_video_tables(unit_of_work: UnitOfWork) -> None:
    async with unit_of_work as uow:
        await uow._session.execute(
            text(
                "TRUNCATE TABLE video_job, video_asset, video_track, "
                "video_tracklet, appearance_interval, video_timeline_chunk, "
                "video_track_sample, process_record, process_event, "
                "idempotency_record, outbox_event "
                "RESTART IDENTITY CASCADE"
            )
        )
        await uow.commit()


@pytest.fixture
async def pending_job(
    unit_of_work: UnitOfWork,
    id_generator: object,
) -> VideoJob:
    video_id = VideoId(id_generator.new_uuid7())
    upload_session_id = UploadSessionId(id_generator.new_uuid7())
    process_id = ProcessId(id_generator.new_uuid7())
    job_id = JobId(id_generator.new_uuid7())

    async with unit_of_work as uow:
        asset = VideoAsset(
            video_id=video_id,
            upload_session_id=upload_session_id,
            state="uploading",
            staging_bucket="test-bucket",
            staging_object_key=f"staging/{video_id}/upload",
        )
        process = ProcessRecord(
            process_id=process_id,
            process_type="video_recognize",
            status="processing",
        )
        job = VideoJob(
            job_id=job_id,
            video_id=video_id,
            process_id=process_id,
            state="pending",
            stage="queued",
            max_attempts=3,
        )
        await uow.video_assets.add(asset)
        await uow.processes.add(process)
        await uow.video_jobs.add(job)
        await uow.commit()
    return job


async def test_claim_next_returns_pending_job(
    unit_of_work: UnitOfWork,
    pending_job: VideoJob,
) -> None:
    now = datetime.now(UTC)
    async with unit_of_work as uow:
        claimed = await uow.video_job_queue.claim_next(
            worker_id="worker-1",
            lease_token=generate_uuid7(),
            now=now,
            lease_expires_at=now + timedelta(seconds=30),
        )
        await uow.commit()

    assert claimed is not None
    assert claimed.job.job_id == pending_job.job_id
    assert claimed.job.state == "processing"
    assert claimed.job.attempt_count == 1
    assert claimed.job.lease_owner == "worker-1"


async def test_concurrent_claim_only_one_winner(
    unit_of_work_factory: UnitOfWorkFactory,
    pending_job: VideoJob,
) -> None:
    results: list[object] = []

    async def claim(worker_id: str) -> None:
        now = datetime.now(UTC)
        async with unit_of_work_factory() as uow:
            claimed = await uow.video_job_queue.claim_next(
                worker_id=worker_id,
                lease_token=generate_uuid7(),
                now=now,
                lease_expires_at=now + timedelta(seconds=30),
            )
            await uow.commit()
        results.append(claimed)

    await asyncio.gather(claim("worker-a"), claim("worker-b"))
    winners = [r for r in results if r is not None]
    losers = [r for r in results if r is None]
    assert len(winners) == 1
    assert len(losers) == 1


async def test_wrong_lease_token_heartbeat_rejected(
    unit_of_work: UnitOfWork,
    pending_job: VideoJob,
) -> None:
    now = datetime.now(UTC)
    async with unit_of_work as uow:
        claimed = await uow.video_job_queue.claim_next(
            worker_id="worker-1",
            lease_token=generate_uuid7(),
            now=now,
            lease_expires_at=now + timedelta(seconds=30),
        )
        await uow.commit()
    assert claimed is not None

    async with unit_of_work as uow:
        ok = await uow.video_job_queue.heartbeat(
            job_id=claimed.job.job_id,
            worker_id="worker-1",
            lease_token=generate_uuid7(),
            expected_version=2,
            now=now + timedelta(seconds=5),
            new_lease_expires_at=now + timedelta(seconds=60),
        )
        await uow.commit()
    assert ok is False


async def test_heartbeat_extends_lease(
    unit_of_work: UnitOfWork,
    pending_job: VideoJob,
) -> None:
    now = datetime.now(UTC)
    token = generate_uuid7()
    async with unit_of_work as uow:
        claimed = await uow.video_job_queue.claim_next(
            worker_id="worker-1",
            lease_token=token,
            now=now,
            lease_expires_at=now + timedelta(seconds=10),
        )
        await uow.commit()
    assert claimed is not None

    async with unit_of_work as uow:
        ok = await uow.video_job_queue.heartbeat(
            job_id=claimed.job.job_id,
            worker_id="worker-1",
            lease_token=token,
                expected_version=claimed.job.version,
                now=now + timedelta(seconds=5),
                new_lease_expires_at=now + timedelta(seconds=60),
        )
        await uow.commit()
    assert ok is True


async def test_update_stage_requires_valid_lease(
    unit_of_work: UnitOfWork,
    pending_job: VideoJob,
) -> None:
    now = datetime.now(UTC)
    token = generate_uuid7()
    async with unit_of_work as uow:
        claimed = await uow.video_job_queue.claim_next(
            worker_id="worker-1",
            lease_token=token,
            now=now,
            lease_expires_at=now + timedelta(seconds=30),
        )
        await uow.commit()
    assert claimed is not None

    async with unit_of_work as uow:
        updated = await uow.video_job_queue.update_stage(
            job_id=claimed.job.job_id,
            worker_id="worker-1",
            lease_token=token,
                expected_version=claimed.job.version,
                stage="decode_infer",
            progress_percent=50,
            processed_frames=10,
            sampled_frames=5,
            detected_observations=3,
            person_count=1,
        )
        await uow.commit()

    assert updated is not None
    assert updated.stage == "decode_infer"
    assert updated.progress_percent == 50


async def test_release_for_retry_reschedules_with_backoff(
    unit_of_work: UnitOfWork,
    pending_job: VideoJob,
) -> None:
    now = datetime.now(UTC)
    token = generate_uuid7()
    async with unit_of_work as uow:
        claimed = await uow.video_job_queue.claim_next(
            worker_id="worker-1",
            lease_token=token,
            now=now,
            lease_expires_at=now + timedelta(seconds=30),
        )
        await uow.commit()
    assert claimed is not None

    retry_at = now + timedelta(seconds=5)
    async with unit_of_work as uow:
        released = await uow.video_job_queue.release_for_retry(
            job_id=claimed.job.job_id,
            worker_id="worker-1",
            lease_token=token,
            available_at=retry_at,
            error_code="transient_decode_error",
        )
        await uow.commit()

    assert released is not None
    assert released.state == "pending"
    assert released.stage == "queued"
    assert released.error_code == "transient_decode_error"
    assert released.attempt_count == 1


async def test_recover_expired_lease_requeues(
    unit_of_work: UnitOfWork,
    pending_job: VideoJob,
) -> None:
    now = datetime.now(UTC)
    token = generate_uuid7()
    async with unit_of_work as uow:
        claimed = await uow.video_job_queue.claim_next(
            worker_id="worker-1",
            lease_token=token,
            now=now,
            lease_expires_at=now + timedelta(seconds=30),
        )
        await uow.commit()
    assert claimed is not None

    async with unit_of_work as uow:
        recovered = await uow.video_job_queue.recover_expired_leases(
            now=now + timedelta(seconds=60),
            batch_size=10,
        )
        await uow.commit()

    assert recovered == 1

    async with unit_of_work as uow:
        job = await uow.video_jobs.get_by_id(claimed.job.job_id)
        assert job is not None
        assert job.state == "pending"
        assert job.lease_owner is None


async def test_cancelling_expired_lease_becomes_cancelled(
    unit_of_work: UnitOfWork,
    pending_job: VideoJob,
) -> None:
    now = datetime.now(UTC)
    token = generate_uuid7()
    async with unit_of_work as uow:
        claimed = await uow.video_job_queue.claim_next(
            worker_id="worker-1",
            lease_token=token,
            now=now,
            lease_expires_at=now + timedelta(seconds=30),
        )
        await uow.video_job_queue.request_cancellation(claimed.job.job_id)
        await uow.commit()

    async with unit_of_work as uow:
        recovered = await uow.video_job_queue.recover_expired_leases(
            now=now + timedelta(seconds=60),
            batch_size=10,
        )
        await uow.commit()

    assert recovered == 1

    async with unit_of_work as uow:
        job = await uow.video_jobs.get_by_id(claimed.job.job_id)
        assert job is not None
        assert job.state == "cancelled"
        assert job.lease_owner is None


async def test_max_attempts_expired_lease_becomes_failed(
    unit_of_work: UnitOfWork,
    id_generator: object,
) -> None:
    video_id = VideoId(id_generator.new_uuid7())
    upload_session_id = UploadSessionId(id_generator.new_uuid7())
    process_id = ProcessId(id_generator.new_uuid7())
    job_id = JobId(id_generator.new_uuid7())

    async with unit_of_work as uow:
        asset = VideoAsset(
            video_id=video_id,
            upload_session_id=upload_session_id,
            state="uploading",
            staging_bucket="test-bucket",
            staging_object_key=f"staging/{video_id}/upload",
        )
        process = ProcessRecord(
            process_id=process_id,
            process_type="video_recognize",
            status="processing",
        )
        job = VideoJob(
            job_id=job_id,
            video_id=video_id,
            process_id=process_id,
            state="pending",
            stage="queued",
            max_attempts=1,
        )
        await uow.video_assets.add(asset)
        await uow.processes.add(process)
        await uow.video_jobs.add(job)
        await uow.commit()

    now = datetime.now(UTC)
    token = generate_uuid7()
    async with unit_of_work as uow:
        claimed = await uow.video_job_queue.claim_next(
            worker_id="worker-1",
            lease_token=token,
            now=now,
            lease_expires_at=now + timedelta(seconds=30),
        )
        await uow.commit()
    assert claimed is not None
    assert claimed.job.attempt_count == 1

    async with unit_of_work as uow:
        recovered = await uow.video_job_queue.recover_expired_leases(
            now=now + timedelta(seconds=60),
            batch_size=10,
        )
        await uow.commit()

    assert recovered == 1

    async with unit_of_work as uow:
        job = await uow.video_jobs.get_by_id(claimed.job.job_id)
        assert job is not None
        assert job.state == "failed"
        assert job.error_code == "max_retry_exceeded"
