"""Tests for the fail-closed test resource guard."""

from __future__ import annotations

import pytest

from tests.support.resource_guard import (
    UnsafeTestResourceError,
    assert_safe_test_environment,
    guard_cleanup,
)


@pytest.fixture
def valid_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MV_TEST_MODE", "1")
    monkeypatch.setenv("MV_TEST_RESOURCE_NAMESPACE", "mergenvision-s01-test")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://mergenvision_test:mergenvision_test@localhost:55432/mergenvision_s01_test",
    )
    monkeypatch.setenv("MINIO_ENDPOINT", "localhost:59000")
    monkeypatch.setenv("MINIO_BUCKET_NAME", "mergenvision-s01-test-face-samples")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:56333")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "mergenvision_s01_test_face_samples_v1")


def test_accepts_valid_test_environment(valid_test_env: None) -> None:
    assert_safe_test_environment()


def test_rejects_missing_test_mode(monkeypatch: pytest.MonkeyPatch, valid_test_env: None) -> None:
    monkeypatch.delenv("MV_TEST_MODE")
    with pytest.raises(UnsafeTestResourceError):
        assert_safe_test_environment()


def test_rejects_wrong_namespace(monkeypatch: pytest.MonkeyPatch, valid_test_env: None) -> None:
    monkeypatch.setenv("MV_TEST_RESOURCE_NAMESPACE", "mergenvision-prod")
    with pytest.raises(UnsafeTestResourceError):
        assert_safe_test_environment()


def test_rejects_wrong_database_name(monkeypatch: pytest.MonkeyPatch, valid_test_env: None) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://mergenvision:mergenvision@localhost:5433/mergenvision",
    )
    with pytest.raises(UnsafeTestResourceError):
        assert_safe_test_environment()


def test_rejects_remote_database_host(
    monkeypatch: pytest.MonkeyPatch, valid_test_env: None
) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://user:pass@postgres.example.com:5432/mergenvision_s01_test",
    )
    with pytest.raises(UnsafeTestResourceError):
        assert_safe_test_environment()


def test_rejects_missing_database_url(
    monkeypatch: pytest.MonkeyPatch, valid_test_env: None
) -> None:
    monkeypatch.delenv("DATABASE_URL")
    with pytest.raises(UnsafeTestResourceError):
        assert_safe_test_environment()


def test_rejects_wrong_minio_bucket(monkeypatch: pytest.MonkeyPatch, valid_test_env: None) -> None:
    monkeypatch.setenv("MINIO_BUCKET_NAME", "mergenvision-face-samples")
    with pytest.raises(UnsafeTestResourceError):
        assert_safe_test_environment()


def test_rejects_remote_minio_endpoint(
    monkeypatch: pytest.MonkeyPatch, valid_test_env: None
) -> None:
    monkeypatch.setenv("MINIO_ENDPOINT", "minio.example.com:9000")
    with pytest.raises(UnsafeTestResourceError):
        assert_safe_test_environment()


def test_rejects_wrong_qdrant_collection(
    monkeypatch: pytest.MonkeyPatch, valid_test_env: None
) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "face_samples_v1")
    with pytest.raises(UnsafeTestResourceError):
        assert_safe_test_environment()


def test_rejects_remote_qdrant_url(monkeypatch: pytest.MonkeyPatch, valid_test_env: None) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.example.com:6333")
    with pytest.raises(UnsafeTestResourceError):
        assert_safe_test_environment()


def test_guard_cleanup_decorator_blocks_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MV_TEST_MODE")
    sentinel = []

    @guard_cleanup
    def cleanup() -> None:
        sentinel.append("ran")

    with pytest.raises(UnsafeTestResourceError):
        cleanup()
    assert not sentinel
