"""Video timeline chunk domain entity."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime


@dataclass
class VideoTimelineChunk:
    chunk_id: uuid.UUID
    job_id: uuid.UUID
    artifact_kind: str
    sequence_no: int
    start_pts_ns: int
    end_pts_ns: int
    bucket: str
    object_key: str
    content_sha256: str
    size_bytes: int
    record_count: int
    schema_version: str
    compression: str
    expires_at: datetime | None = None
    created_at: datetime | None = None
