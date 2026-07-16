"""Pydantic-settings based configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+asyncpg://mergenvision:mergenvision@localhost:5433/mergenvision"

    minio_endpoint: str = "localhost:9002"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_bucket_name: str = "mergenvision-face-samples"

    qdrant_url: str = "http://localhost:6335"
    qdrant_collection_name: str = "face_samples_v1"

    log_level: str = "INFO"


settings = Settings()
