"""Fail-closed guard for test resource isolation.

Acceptance tests must never operate against development, staging or production
resources. This module validates that every required environment variable
points to the dedicated Sprint 01 test namespace and localhost endpoints
before any cleanup or mutation is performed.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse


class UnsafeTestResourceError(Exception):
    """Raised when the test environment is not safely isolated."""


_REQUIRED_ENV = {
    "MV_TEST_MODE": "1",
    "MV_TEST_RESOURCE_NAMESPACE": "mergenvision-s01-test",
    "MINIO_BUCKET_NAME": "mergenvision-s01-test-face-samples",
    "QDRANT_COLLECTION_NAME": "mergenvision_s01_test_face_samples_v1",
}

_REQUIRED_DB_NAME = "mergenvision_s01_test"
_LOCAL_HOSTS = {"localhost", "127.0.0.1"}


def _require_exact(name: str, expected: str) -> None:
    value = os.environ.get(name)
    if value != expected:
        raise UnsafeTestResourceError(f"{name} must be exactly {expected!r}, got {value!r}")


def _require_database_local() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise UnsafeTestResourceError("DATABASE_URL is not set")

    parsed = urlparse(url)
    db_name = parsed.path.lstrip("/").split("?")[0]
    if db_name != _REQUIRED_DB_NAME:
        raise UnsafeTestResourceError(
            f"DATABASE_URL database must be {_REQUIRED_DB_NAME!r}, got {db_name!r}"
        )

    host = parsed.hostname
    if host not in _LOCAL_HOSTS:
        raise UnsafeTestResourceError(
            f"DATABASE_URL host must be localhost/127.0.0.1, got {host!r}"
        )


def _require_minio_local() -> None:
    endpoint = os.environ.get("MINIO_ENDPOINT", "")
    host = endpoint.split(":")[0]
    if host not in _LOCAL_HOSTS:
        raise UnsafeTestResourceError(
            f"MINIO_ENDPOINT host must be localhost/127.0.0.1, got {host!r}"
        )


def _require_qdrant_local() -> None:
    url = os.environ.get("QDRANT_URL", "")
    parsed = urlparse(url)
    host = parsed.hostname
    if host not in _LOCAL_HOSTS:
        raise UnsafeTestResourceError(f"QDRANT_URL host must be localhost/127.0.0.1, got {host!r}")


def assert_safe_test_environment() -> None:
    """Validate that the process is configured for the dedicated test namespace.

    Raises UnsafeTestResourceError on any mismatch.
    """
    for name, expected in _REQUIRED_ENV.items():
        _require_exact(name, expected)

    _require_database_local()
    _require_minio_local()
    _require_qdrant_local()


def guard_cleanup(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that runs assert_safe_test_environment before cleanup code."""

    def wrapper(*args: object, **kwargs: object) -> Any:
        assert_safe_test_environment()
        return func(*args, **kwargs)

    return wrapper
