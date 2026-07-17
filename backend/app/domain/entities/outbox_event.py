"""Outbox event domain entity for cross-store/out-process notifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID


@dataclass
class OutboxEvent:
    outbox_event_id: UUID
    aggregate_type: str
    aggregate_id: UUID
    event_type: str
    dedupe_key: str
    state: str
    attempt_count: int
    max_attempts: int
    available_at: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    locked_by: str | None = None
    locked_until: datetime | None = None
    last_error_code: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    succeeded_at: datetime | None = None
    failed_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.state not in {"pending", "processing", "succeeded", "failed", "dead_letter"}:
            raise ValueError(f"Invalid outbox state: {self.state}")
        if self.attempt_count < 0:
            raise ValueError("attempt_count must be non-negative")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
