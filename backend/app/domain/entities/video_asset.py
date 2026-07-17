"""Video asset domain entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.domain.value_objects import UploadSessionId, VideoId


@dataclass
class VideoAsset:
    video_id: VideoId
    upload_session_id: UploadSessionId
    state: str = "uploading"
    staging_bucket: str | None = None
    staging_object_key: str | None = None
    bucket: str | None = None
    object_key: str | None = None
    content_sha256: str | None = None
    size_bytes: int | None = None
    content_type: str | None = None
    container_format: str | None = None
    video_codec: str | None = None
    pixel_format: str | None = None
    display_width: int | None = None
    display_height: int | None = None
    rotation_degrees: int = 0
    duration_ns: int | None = None
    time_base_num: int | None = None
    time_base_den: int | None = None
    nominal_fps_num: int | None = None
    nominal_fps_den: int | None = None
    total_frames: int | None = None
    retention_until: datetime | None = None
    failure_code: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ready_at: datetime | None = None
    deleted_at: datetime | None = None
    version: int = 1

    def mark_ready(
        self,
        *,
        bucket: str,
        object_key: str,
        content_sha256: str,
        size_bytes: int,
        content_type: str,
        container_format: str,
        video_codec: str,
        pixel_format: str | None,
        display_width: int,
        display_height: int,
        rotation_degrees: int,
        duration_ns: int,
        time_base_num: int,
        time_base_den: int,
        nominal_fps_num: int | None,
        nominal_fps_den: int | None,
        total_frames: int | None,
        retention_until: datetime,
    ) -> None:
        if not all(
            [
                bucket,
                object_key,
                content_sha256,
                size_bytes >= 0,
                container_format,
                video_codec,
                display_width > 0,
                display_height > 0,
                duration_ns >= 0,
                time_base_den > 0,
            ]
        ):
            raise ValueError("VideoAsset ready fields incomplete")
        self.state = "ready"
        self.bucket = bucket
        self.object_key = object_key
        self.content_sha256 = content_sha256
        self.size_bytes = size_bytes
        self.content_type = content_type
        self.container_format = container_format
        self.video_codec = video_codec
        self.pixel_format = pixel_format
        self.display_width = display_width
        self.display_height = display_height
        self.rotation_degrees = rotation_degrees
        self.duration_ns = duration_ns
        self.time_base_num = time_base_num
        self.time_base_den = time_base_den
        self.nominal_fps_num = nominal_fps_num
        self.nominal_fps_den = nominal_fps_den
        self.total_frames = total_frames
        self.retention_until = retention_until
        self.ready_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)
        self.version += 1

    def mark_rejected(self, failure_code: str) -> None:
        self.state = "rejected"
        self.failure_code = failure_code
        self.updated_at = datetime.now(UTC)
        self.version += 1
