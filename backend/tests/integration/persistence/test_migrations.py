"""Integration tests for Alembic migration and schema."""

from __future__ import annotations

import os
from typing import Any

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine

pytestmark = pytest.mark.asyncio(scope="session")


def _check_sql(constraint: dict[str, Any]) -> str:
    return constraint.get("sqltext") or constraint.get("sql") or ""


async def test_upgrade_head_creates_tables() -> None:
    engine = create_async_engine(os.environ["DATABASE_URL"])
    async with engine.connect() as conn:
        tables = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())
        assert "face_identity" in tables
        assert "face_sample" in tables
        assert "process_record" in tables
        assert "recognition_result" in tables
    await engine.dispose()


async def test_required_constraints_and_indexes() -> None:
    engine = create_async_engine(os.environ["DATABASE_URL"])

    def _reflect_schema(sync_conn: Any) -> dict[str, Any]:
        inspector = inspect(sync_conn)
        return {
            "identity_indexes": {
                idx["name"]: idx for idx in inspector.get_indexes("face_identity")
            },
            "process_indexes": {
                idx["name"]: idx for idx in inspector.get_indexes("process_record")
            },
            "sample_indexes": {idx["name"]: idx for idx in inspector.get_indexes("face_sample")},
            "result_indexes": {
                idx["name"]: idx for idx in inspector.get_indexes("recognition_result")
            },
            "identity_checks": inspector.get_check_constraints("face_identity"),
            "process_checks": inspector.get_check_constraints("process_record"),
            "sample_checks": inspector.get_check_constraints("face_sample"),
            "result_checks": inspector.get_check_constraints("recognition_result"),
            "sample_fks": inspector.get_foreign_keys("face_sample"),
            "result_fks": inspector.get_foreign_keys("recognition_result"),
        }

    async with engine.connect() as conn:
        schema = await conn.run_sync(_reflect_schema)

    # Indexes on face_identity
    assert "face_identity_status_is_active_idx" in schema["identity_indexes"]
    assert "face_identity_created_at_idx" in schema["identity_indexes"]

    # Indexes on process_record
    assert "process_record_status_created_at_idx" in schema["process_indexes"]
    assert "process_record_process_type_created_at_idx" in schema["process_indexes"]

    # Indexes on face_sample
    assert "face_sample_face_id_sample_state_idx" in schema["sample_indexes"]
    assert "face_sample_bucket_key_unique_idx" in schema["sample_indexes"]
    assert schema["sample_indexes"]["face_sample_bucket_key_unique_idx"]["unique"] is True

    # Indexes on recognition_result
    assert "recognition_result_process_id_result_index_idx" in schema["result_indexes"]

    # Check constraints
    assert any(
        "anonymous" in _check_sql(c) and "known" in _check_sql(c) for c in schema["identity_checks"]
    )
    assert any("processing" in _check_sql(c) for c in schema["process_checks"])
    assert any("pending" in _check_sql(c) for c in schema["sample_checks"])
    assert any("new_anonymous" in _check_sql(c) for c in schema["result_checks"])

    # Foreign keys
    assert any(fk["referred_table"] == "face_identity" for fk in schema["sample_fks"])
    referred = {fk["referred_table"] for fk in schema["result_fks"]}
    assert "face_identity" in referred
    assert "process_record" in referred
    assert "face_sample" in referred

    await engine.dispose()
