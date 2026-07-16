"""Unit tests for UUIDv7 generator."""

import uuid

from app.infrastructure.uuid7 import generate_uuid7


def test_generate_uuid7_returns_uuid_instance() -> None:
    value = generate_uuid7()
    assert isinstance(value, uuid.UUID)


def test_generate_uuid7_version_is_7() -> None:
    value = generate_uuid7()
    assert value.version == 7


def test_generate_uuid7_produces_distinct_values() -> None:
    values = {generate_uuid7() for _ in range(100)}
    assert len(values) == 100
