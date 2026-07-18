"""Phase 1 bulk enrollment configuration.

Security rule: no hardcoded secrets or shared-storage addresses. All connection
strings and credentials come from environment variables.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any, cast

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # Required secrets / addresses
    database_url: str = Field(..., alias="DATABASE_URL")
    minio_endpoint: str = Field(..., alias="MV_MINIO_ENDPOINT")
    minio_access_key: str = Field(..., alias="MV_MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(..., alias="MV_MINIO_SECRET_KEY")
    minio_secure: bool = Field(default=False, alias="MV_MINIO_SECURE")
    minio_bucket_name: str = Field(..., alias="MV_MINIO_BUCKET_NAME")
    qdrant_url: str = Field(..., alias="MV_QDRANT_URL")
    qdrant_collection_name: str = Field(
        default="face_samples_retinaface_r50_glintr100_v1",
        alias="MV_QDRANT_COLLECTION_NAME",
    )
    id_hmac_key: str = Field(..., alias="MV_PHASE1_BULK_ID_HMAC_KEY")

    # Model / runtime
    model_version: str = Field(
        default="retinaface_r50_glintr100_v1",
        alias="MV_MODEL_VERSION",
    )
    model_profile_path: str = Field(
        default="config/model_profile.json",
        alias="MV_MODEL_PROFILE_PATH",
    )

    # GPU tuning
    gpu_device_id: int = Field(default=0, alias="MV_GPU_DEVICE_ID")
    inference_slot_count: int = Field(default=1, alias="MV_INFERENCE_SLOT_COUNT")
    bulk_extract_batch_size: int = Field(default=16, alias="MV_BULK_EXTRACT_BATCH_SIZE")
    bulk_max_persistence_concurrency: int = Field(default=32, alias="MV_BULK_MAX_PERSISTENCE_CONCURRENCY")
    bulk_activation_batch_size: int = Field(default=2048, alias="MV_BULK_ACTIVATION_BATCH_SIZE")
    recognizer_max_faces_per_batch: int = Field(default=256, alias="MV_RECOGNIZER_MAX_FACES_PER_BATCH")
    match_threshold: float = Field(default=0.55, alias="MV_MATCH_THRESHOLD")
    max_image_bytes: int = Field(default=50 * 1024 * 1024, alias="MV_MAX_IMAGE_BYTES")
    max_image_width: int = Field(default=8192, alias="MV_MAX_IMAGE_WIDTH")
    max_image_height: int = Field(default=8192, alias="MV_MAX_IMAGE_HEIGHT")
    max_image_pixels: int = Field(default=67_108_864, alias="MV_MAX_IMAGE_PIXELS")

    log_level: str = Field(default="INFO", alias="MV_LOG_LEVEL")

    @property
    def model_profile_file(self) -> Path:
        return Path(self.model_profile_path)


def load_model_profile(
    profile_path: Path | str,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Load model profile and resolve relative engine paths.

    Relative engine paths are resolved against ``repo_root`` (or the profile's
    parent directory when ``repo_root`` is not provided).
    """
    profile_path = Path(profile_path).resolve()
    repo_root = profile_path.parent if repo_root is None else Path(repo_root).resolve()

    with profile_path.open("r", encoding="utf-8") as f:
        profile = cast(dict[str, Any], json.load(f))

    for key in ("retinaface_r50_dynamic", "glintr100"):
        entry = profile["engine_manifest"][key]
        engine_path = Path(entry["engine_path"])
        if not engine_path.is_absolute():
            entry["engine_path"] = str(repo_root / engine_path)

    return profile


@cache
def get_settings() -> Settings:
    """Load settings lazily so unit tests can import the module without env vars."""
    return Settings()  # type: ignore[call-arg]


# Backwards-compatible alias retained for existing call sites.
settings = get_settings
