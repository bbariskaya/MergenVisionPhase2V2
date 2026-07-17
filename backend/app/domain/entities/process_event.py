"""Process event domain entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID


@dataclass
class ProcessEvent:
    event_id: UUID
    process_id: UUID
    sequence_no: int
    event_type: str
    severity: str
    payload: dict[str, Any] = field(default_factory=dict)
    job_id: UUID | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.severity not in {"info", "warning", "error"}:
            raise ValueError(f"Invalid severity: {self.severity}")
