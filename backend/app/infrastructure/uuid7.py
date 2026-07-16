"""UUIDv7 generation implementation."""

from __future__ import annotations

import uuid

from uuid_extensions import uuid7

from app.application.ports.id_generator import IdGenerator


def generate_uuid7() -> uuid.UUID:
    """Return a new UUIDv7 as a standard uuid.UUID."""
    return uuid.UUID(str(uuid7()))


class Uuid7Generator(IdGenerator):
    def new_uuid7(self) -> uuid.UUID:
        return generate_uuid7()
