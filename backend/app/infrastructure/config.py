"""Pydantic-settings based configuration.

Required values are supplied through environment variables or an `.env` file.
No secret defaults are baked into the source; missing required settings cause
an explicit validation failure at startup.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    database_url: str

    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool = False
    minio_bucket_name: str

    qdrant_url: str
    qdrant_collection_name: str

    model_version: str

    # Native GPU inference configuration
    model_profile_path: str
    detector_engine_path: str
    recognizer_engine_path: str
    gpu_device_id: int = 0
    inference_slot_count: int = 1
    match_threshold: float = 0.55
    max_image_bytes: int = 25 * 1024 * 1024
    max_image_width: int = 8192
    max_image_height: int = 8192
    max_image_pixels: int = 67_108_864

    log_level: str = "INFO"

    # Video upload configuration
    max_video_bytes: int = 2 * 1024 * 1024 * 1024
    max_video_duration_ns: int = 600 * 1_000_000_000
    max_video_display_width: int = 7680
    max_video_display_height: int = 4320
    allowed_video_containers: list[str] = ["mp4", "mov", "mkv"]
    allowed_video_codecs: list[str] = ["h264", "hevc"]
    video_retention_seconds: int = 7 * 24 * 3600
    ffprobe_command: list[str] = ["ffprobe"]
    video_staging_prefix: str = "staging/videos/"
    video_source_prefix: str = "videos/"
    video_temp_dir: str | None = None
    video_probe_timeout_seconds: float = 60.0
    video_max_attempts: int = 3


settings = Settings()  # type: ignore[call-arg]
