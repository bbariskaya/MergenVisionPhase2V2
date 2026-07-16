"""Port for generating persistent business identifiers."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod


class IdGenerator(ABC):
    @abstractmethod
    def new_uuid7(self) -> uuid.UUID:
        """Return a new UUIDv7 as a standard uuid.UUID."""
        ...
