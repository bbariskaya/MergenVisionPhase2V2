"""UUIDv7 generation wrapper."""

from __future__ import annotations

import uuid

from uuid_extensions import uuid7


def generate_uuid7() -> uuid.UUID:
    """Return a new UUIDv7 as a standard uuid.UUID."""
    return uuid.UUID(str(uuid7()))
