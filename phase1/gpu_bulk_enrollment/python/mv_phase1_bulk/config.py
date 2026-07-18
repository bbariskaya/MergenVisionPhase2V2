"""Phase 1 bulk enrollment configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    database_url: str = "postgresql+asyncpg://user:pass@localhost/mergenvision"

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_bucket_name: str = "mergenvision"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "faces"

    model_version: str = "retinaface_r50_glintr100_v1"
    model_profile_path: str = "config/model_profile.json"

    # Phase 1 engine paths (overridden by model_profile engine_manifest).
    detector_engine_path: str = ""
    recognizer_engine_path: str = ""

    gpu_device_id: int = 0
    inference_slot_count: int = 1
    bulk_extract_batch_size: int = 256
    bulk_max_persistence_concurrency: int = 32
    bulk_activation_batch_size: int = 2048
    recognizer_max_faces_per_batch: int = 256
    match_threshold: float = 0.55
    max_image_bytes: int = 25 * 1024 * 1024
    max_image_width: int = 8192
    max_image_height: int = 8192
    max_image_pixels: int = 67_108_864

    log_level: str = "INFO"

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
    if repo_root is None:
        repo_root = profile_path.parent
    else:
        repo_root = Path(repo_root).resolve()

    with profile_path.open("r", encoding="utf-8") as f:
        profile = json.load(f)

    for key in ("retinaface_r50_dynamic", "glintr100"):
        entry = profile["engine_manifest"][key]
        engine_path = Path(entry["engine_path"])
        if not engine_path.is_absolute():
            entry["engine_path"] = str(repo_root / engine_path)

    return profile


settings = Settings()
