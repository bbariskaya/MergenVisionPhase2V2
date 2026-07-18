"""Admission gate: validate environment and service connectivity before a run."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from sqlalchemy import text

from mv_phase1_bulk.config import Settings


@dataclass
class AdmissionReport:
    """Result of an admission check."""

    ready: bool
    problems: list[str] = field(default_factory=list)

    def raise_if_not_ready(self) -> None:
        if not self.ready:
            raise AdmissionError("; ".join(self.problems))


class AdmissionError(RuntimeError):
    """Raised when the admission gate blocks a run."""


class AdmissionGate:
    """Fail-fast checks for required environment variables and services."""

    REQUIRED_ENV_VARS = [
        "DATABASE_URL",
        "MV_MINIO_ENDPOINT",
        "MV_MINIO_ACCESS_KEY",
        "MV_MINIO_SECRET_KEY",
        "MV_MINIO_BUCKET_NAME",
        "MV_QDRANT_URL",
        "MV_PHASE1_BULK_ID_HMAC_KEY",
    ]

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def check_env(self) -> AdmissionReport:
        """Validate that all required environment variables are non-empty."""
        problems: list[str] = []
        for var in self.REQUIRED_ENV_VARS:
            value = os.environ.get(var, "")
            if not value:
                problems.append(f"missing_or_empty:{var}")
        return AdmissionReport(ready=not problems, problems=problems)

    async def check_minio(self) -> AdmissionReport:
        """Validate MinIO connectivity by checking bucket existence."""
        from mv_phase1_bulk.minio_store import MinioStore

        store = MinioStore(
            endpoint=self._settings.minio_endpoint,
            access_key=self._settings.minio_access_key,
            secret_key=self._settings.minio_secret_key,
            bucket_name=self._settings.minio_bucket_name,
            secure=self._settings.minio_secure,
        )
        try:
            await store._ensure_bucket()
            return AdmissionReport(ready=True)
        except Exception as exc:  # noqa: BLE001
            return AdmissionReport(ready=False, problems=[f"minio_unavailable:{type(exc).__name__}:{exc}"])

    async def check_qdrant(self) -> AdmissionReport:
        """Validate Qdrant connectivity by fetching collection info."""
        from mv_phase1_bulk.qdrant_store import QdrantStore

        store = QdrantStore(
            url=self._settings.qdrant_url,
            collection_name=self._settings.qdrant_collection_name,
            model_version=self._settings.model_version,
        )
        try:
            await store.ensure_collection()
            return AdmissionReport(ready=True)
        except Exception as exc:  # noqa: BLE001
            return AdmissionReport(ready=False, problems=[f"qdrant_unavailable:{type(exc).__name__}:{exc}"])

    async def check_postgres(self) -> AdmissionReport:
        """Validate PostgreSQL connectivity with a lightweight query."""
        from mv_phase1_bulk.postgres_store import PostgresStore

        store = PostgresStore(self._settings.database_url)
        try:
            await store.connect()
            async with store.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return AdmissionReport(ready=True)
        except Exception as exc:  # noqa: BLE001
            return AdmissionReport(ready=False, problems=[f"postgres_unavailable:{type(exc).__name__}:{exc}"])
        finally:
            await store.close()

    async def admit(self, *, check_services: bool = True) -> AdmissionReport:
        """Run all checks and return a combined report."""
        report = self.check_env()
        if not report.ready:
            return report

        if check_services:
            for coro in (self.check_postgres(), self.check_minio(), self.check_qdrant()):
                sub = await coro
                report.problems.extend(sub.problems)
            report.ready = not report.problems

        return report
