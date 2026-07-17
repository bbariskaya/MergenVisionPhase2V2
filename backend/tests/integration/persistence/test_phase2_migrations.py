"""Failing schema-contract tests for Phase 2 video control-plane migrations."""

from __future__ import annotations

import os
from typing import Any

import asyncpg
import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine

from app.infrastructure.uuid7 import generate_uuid7

pytestmark = pytest.mark.asyncio(scope="session")


def _pg_url() -> str:
    return os.environ["DATABASE_URL"].replace("postgresql+asyncpg", "postgresql")


def _check_sql(constraint: dict[str, Any]) -> str:
    return constraint.get("sqltext") or constraint.get("sql") or ""


async def test_phase2_video_tables_exist() -> None:
    engine = create_async_engine(os.environ["DATABASE_URL"])
    async with engine.connect() as conn:
        tables = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())
        for expected in (
            "video_asset",
            "video_job",
            "idempotency_record",
            "process_event",
            "outbox_event",
            "video_track",
            "video_tracklet",
            "appearance_interval",
            "video_timeline_chunk",
            "video_track_sample",
        ):
            assert expected in tables, f"missing table {expected}"
    await engine.dispose()


async def test_process_record_allows_cancelled_and_video_recognize() -> None:
    engine = create_async_engine(os.environ["DATABASE_URL"])

    def _reflect(sync_conn: Any) -> dict[str, Any]:
        inspector = inspect(sync_conn)
        return {
            "checks": inspector.get_check_constraints("process_record"),
        }

    async with engine.connect() as conn:
        schema = await conn.run_sync(_reflect)

    checks = " ".join(_check_sql(c) for c in schema["checks"])
    assert "cancelled" in checks
    assert "video_recognize" in checks
    await engine.dispose()


async def test_video_asset_constraints_and_indexes() -> None:
    engine = create_async_engine(os.environ["DATABASE_URL"])

    def _reflect(sync_conn: Any) -> dict[str, Any]:
        inspector = inspect(sync_conn)
        return {
            "checks": inspector.get_check_constraints("video_asset"),
            "indexes": {idx["name"]: idx for idx in inspector.get_indexes("video_asset")},
            "columns": inspector.get_columns("video_asset"),
        }

    async with engine.connect() as conn:
        schema = await conn.run_sync(_reflect)

    assert any("size_bytes" in _check_sql(c) for c in schema["checks"])
    assert any("duration_ns" in _check_sql(c) for c in schema["checks"])
    assert "video_asset_bucket_key_partial_idx" in schema["indexes"]
    assert "video_asset_retention_idx" in schema["indexes"]
    await engine.dispose()


async def test_video_job_constraints_and_indexes() -> None:
    engine = create_async_engine(os.environ["DATABASE_URL"])

    def _reflect(sync_conn: Any) -> dict[str, Any]:
        inspector = inspect(sync_conn)
        return {
            "checks": inspector.get_check_constraints("video_job"),
            "indexes": {idx["name"]: idx for idx in inspector.get_indexes("video_job")},
            "fks": inspector.get_foreign_keys("video_job"),
        }

    async with engine.connect() as conn:
        schema = await conn.run_sync(_reflect)

    checks = " ".join(_check_sql(c) for c in schema["checks"])
    assert "progress_percent" in checks
    assert "attempt_count" in checks
    assert "state" in checks
    assert "completed_at" in checks
    assert "video_job_pending_claim_idx" in schema["indexes"]
    assert "video_job_lease_recovery_idx" in schema["indexes"]

    referred = {fk["referred_table"] for fk in schema["fks"]}
    assert "video_asset" in referred
    assert "process_record" in referred
    await engine.dispose()


async def test_video_result_tables_constraints_and_indexes() -> None:
    engine = create_async_engine(os.environ["DATABASE_URL"])

    def _reflect(sync_conn: Any) -> dict[str, Any]:
        inspector = inspect(sync_conn)
        return {
            "track_checks": inspector.get_check_constraints("video_track"),
            "track_indexes": {idx["name"]: idx for idx in inspector.get_indexes("video_track")},
            "tracklet_indexes": {idx["name"]: idx for idx in inspector.get_indexes("video_tracklet")},
            "appearance_indexes": {idx["name"]: idx for idx in inspector.get_indexes("appearance_interval")},
            "timeline_indexes": {idx["name"]: idx for idx in inspector.get_indexes("video_timeline_chunk")},
            "sample_indexes": {idx["name"]: idx for idx in inspector.get_indexes("video_track_sample")},
        }

    async with engine.connect() as conn:
        schema = await conn.run_sync(_reflect)

    assert "video_track_job_ordinal_idx" in schema["track_indexes"]
    assert "video_tracklet_job_ordinal_idx" in schema["tracklet_indexes"]
    assert "appearance_track_interval_idx" in schema["appearance_indexes"]
    assert "video_timeline_chunk_job_kind_seq_idx" in schema["timeline_indexes"]
    assert "video_timeline_chunk_bucket_key_unique_idx" in schema["timeline_indexes"]
    assert "video_track_sample_track_rank_idx" in schema["sample_indexes"]

    track_checks = " ".join(_check_sql(c) for c in schema["track_checks"])
    assert "match_confidence" in track_checks
    await engine.dispose()


async def _assert_rejected(
    conn: asyncpg.Connection,
    sql: str,
    *args: object,
    setup_sqls: list[tuple[str, tuple[object, ...]]] | None = None,
) -> None:
    txn = conn.transaction()
    await txn.start()
    try:
        if setup_sqls:
            for setup_sql, setup_args in setup_sqls:
                await conn.execute(setup_sql, *setup_args)
        with pytest.raises(asyncpg.exceptions.IntegrityConstraintViolationError):
            await conn.execute(sql, *args)
    finally:
        await txn.rollback()


async def test_phase2_invalid_inserts_rejected() -> None:
    video_id = generate_uuid7()
    process_id = generate_uuid7()
    conn = await asyncpg.connect(_pg_url())
    try:
        # invalid video_asset state
        await _assert_rejected(
            conn,
            "INSERT INTO video_asset (video_id, upload_session_id, state) VALUES ($1, $2, 'magic')",
            generate_uuid7(),
            generate_uuid7(),
        )

        # pending job cannot hold an active lease
        await _assert_rejected(
            conn,
            "INSERT INTO video_job (job_id, video_id, process_id, state, stage, sampling_mode, available_at, max_attempts, "
            "lease_owner, lease_expires_at) VALUES ($1, $2, $3, 'pending', 'queued', 'every_frame', now(), 3, 'worker-1', now())",
            generate_uuid7(),
            video_id,
            process_id,
            setup_sqls=[
                (
                    "INSERT INTO video_asset (video_id, upload_session_id, state) VALUES ($1, $2, 'uploading')",
                    (video_id, generate_uuid7()),
                ),
                (
                    "INSERT INTO process_record (process_id, process_type, status) VALUES ($1, 'video_recognize', 'processing')",
                    (process_id,),
                ),
            ],
        )

        # failed job requires error_code and failed_at
        await _assert_rejected(
            conn,
            "INSERT INTO video_job (job_id, video_id, process_id, state, stage, sampling_mode, available_at, max_attempts, "
            "failed_at) VALUES ($1, $2, $3, 'failed', 'finalize', 'every_frame', now(), 3, now())",
            generate_uuid7(),
            video_id,
            process_id,
            setup_sqls=[
                (
                    "INSERT INTO video_asset (video_id, upload_session_id, state) VALUES ($1, $2, 'uploading')",
                    (video_id, generate_uuid7()),
                ),
                (
                    "INSERT INTO process_record (process_id, process_type, status) VALUES ($1, 'video_recognize', 'processing')",
                    (process_id,),
                ),
            ],
        )

        # invalid sampling mode combination: every_n_frames with frames_per_second set
        await _assert_rejected(
            conn,
            "INSERT INTO video_job (job_id, video_id, process_id, state, stage, sampling_mode, available_at, max_attempts, "
            "every_n_frames, frames_per_second) VALUES ($1, $2, $3, 'pending', 'queued', 'every_n_frames', now(), 3, 5, 1.0)",
            generate_uuid7(),
            video_id,
            process_id,
            setup_sqls=[
                (
                    "INSERT INTO video_asset (video_id, upload_session_id, state) VALUES ($1, $2, 'uploading')",
                    (video_id, generate_uuid7()),
                ),
                (
                    "INSERT INTO process_record (process_id, process_type, status) VALUES ($1, 'video_recognize', 'processing')",
                    (process_id,),
                ),
            ],
        )
    finally:
        await conn.close()
