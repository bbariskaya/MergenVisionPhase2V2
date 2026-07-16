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

    log_level: str = "INFO"


settings = Settings()  # type: ignore[call-arg]
