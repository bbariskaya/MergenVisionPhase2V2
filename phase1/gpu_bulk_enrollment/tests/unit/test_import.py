"""Static import/version checks for Phase 1 package."""

from typing import Any

import mv_phase1_bulk
from mv_phase1_bulk import cli, config, manifest


def test_version() -> None:
    assert mv_phase1_bulk.__version__ == "0.1.0"


def test_cli_app_exists() -> None:
    assert cli.app is not None


def test_settings_load(monkeypatch: Any) -> None:
    config.get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
    monkeypatch.setenv("MV_MINIO_ENDPOINT", "localhost:9000")
    monkeypatch.setenv("MV_MINIO_ACCESS_KEY", "test")
    monkeypatch.setenv("MV_MINIO_SECRET_KEY", "test")
    monkeypatch.setenv("MV_MINIO_BUCKET_NAME", "test")
    monkeypatch.setenv("MV_QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("MV_PHASE1_BULK_ID_HMAC_KEY", "test-key")
    assert config.get_settings().model_version == "retinaface_r50_glintr100_v1"


def test_manifest_class() -> None:
    assert manifest.EnrollmentManifest is not None


def test_pipeline_class() -> None:
    pytest = __import__("pytest")
    pytest.importorskip("cuda.bindings")
    from mv_phase1_bulk import pipeline

    assert pipeline.GpuFacePipeline is not None
