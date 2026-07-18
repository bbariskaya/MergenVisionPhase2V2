"""Tests for the local run journal and GPU-free CLI commands."""

from __future__ import annotations

from typing import Any

import pytest
import typer
from mv_phase1_bulk.cli import RunJournal, _load_run_journal, app
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
    monkeypatch.setenv("MV_MINIO_ENDPOINT", "localhost:9000")
    monkeypatch.setenv("MV_MINIO_ACCESS_KEY", "test")
    monkeypatch.setenv("MV_MINIO_SECRET_KEY", "test")
    monkeypatch.setenv("MV_MINIO_BUCKET_NAME", "test")
    monkeypatch.setenv("MV_QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("MV_PHASE1_BULK_ID_HMAC_KEY", "test-key")


def test_run_journal_round_trip(tmp_path: Any) -> None:
    journal = RunJournal(
        run_id="r1",
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:01:00+00:00",
        source_namespace="ns",
        dataset_root="/data",
        manifest="/data/manifest.jsonl",
        model_version="mv1",
        preprocess_version="pv1",
        gpu_device=0,
        batch_size=16,
        hmac_fingerprint="abc",
        outcomes=[
            {
                "external_subject_key": "alice",
                "person_id": str(__import__("uuid").uuid4()),
                "face_id": str(__import__("uuid").uuid4()),
                "persisted_sample_ids": [str(__import__("uuid").uuid4())],
                "failed_sample_ids": [],
                "errors": [],
            }
        ],
    )
    path = tmp_path / "r1.json"
    path.write_text(journal.to_json(), encoding="utf-8")
    loaded = RunJournal.from_file(path)
    assert loaded.run_id == journal.run_id
    assert loaded.outcomes[0]["external_subject_key"] == "alice"


def test_load_run_journal_missing_raises() -> None:
    with pytest.raises(typer.BadParameter):
        _load_run_journal("does-not-exist")


def test_report_command_requires_existing_run() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["report", "--run-id", "missing"])
    assert result.exit_code != 0


def test_reconcile_command_requires_existing_run() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["reconcile", "--run-id", "missing"])
    assert result.exit_code != 0
