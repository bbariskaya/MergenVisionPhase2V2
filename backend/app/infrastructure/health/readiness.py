"""Readiness probing for external dependencies and the native GPU runtime."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import text

from app.infrastructure.config import Settings, settings
from app.infrastructure.model_profile import ModelProfile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    ready: bool
    message: str
    retryable: bool = False


@dataclass(frozen=True)
class ReadinessReport:
    ready: bool
    status: str
    message: str
    checks: tuple[ReadinessCheck, ...] = field(default_factory=tuple)


class ReadinessProbe(Protocol):
    async def check(self) -> ReadinessReport:
        ...


class DefaultReadinessProbe:
    """Readiness probe backed by real dependency clients.

    Note: the probe intentionally does **not** build the native engine at
    startup.  Native initialization is performed once by the application
    lifespan and is reported separately so that a dependency outage does not
    silently block the event loop.
    """

    def __init__(
        self,
        settings_obj: Settings | None = None,
        session_maker: Any | None = None,
        minio_store: Any | None = None,
        qdrant_store: Any | None = None,
    ) -> None:
        self._settings = settings_obj or settings
        self._session_maker = session_maker
        self._minio_store = minio_store
        self._qdrant_store = qdrant_store

    async def check(self) -> ReadinessReport:
        checks: list[ReadinessCheck] = []
        results = await asyncio.gather(
            self._check_configuration(),
            self._check_database(),
            self._check_minio(),
            self._check_qdrant(),
            self._check_native_module(),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                checks.append(
                    ReadinessCheck(
                        name="unknown",
                        ready=False,
                        message=f"probe crashed: {result!s}",
                        retryable=True,
                    )
                )
            else:
                checks.append(result)

        ready = all(c.ready for c in checks)
        return ReadinessReport(
            ready=ready,
            status="ok" if ready else "not_ready",
            message="".join(c.message for c in checks if not c.ready) or "ok",
            checks=tuple(checks),
        )

    async def _check_configuration(self) -> ReadinessCheck:
        required_paths = [
            ("model_profile", self._settings.model_profile_path),
            ("detector_engine", self._settings.detector_engine_path),
            ("recognizer_engine", self._settings.recognizer_engine_path),
        ]
        try:
            profile = ModelProfile.load(self._settings.model_profile_path)
        except FileNotFoundError as exc:
            return ReadinessCheck(
                name="configuration",
                ready=False,
                message=f"model profile missing: {exc}",
            )
        except Exception as exc:
            return ReadinessCheck(
                name="configuration",
                ready=False,
                message=f"model profile invalid: {exc}",
            )

        missing = []
        for label, path in required_paths:
            if not Path(path).exists():
                missing.append(label)
        if missing:
            return ReadinessCheck(
                name="configuration",
                ready=False,
                message=f"missing artifacts: {', '.join(missing)}",
            )

        return ReadinessCheck(
            name="configuration",
            ready=True,
            message=f"model_version={profile.model_version}, preprocess={profile.preprocess_version}",
        )

    async def _check_database(self) -> ReadinessCheck:
        if self._session_maker is None:
            return ReadinessCheck(
                name="database",
                ready=False,
                message="session maker not configured",
                retryable=True,
            )
        try:
            async with self._session_maker() as session:
                await session.execute(text("SELECT 1"))
        except Exception as exc:
            return ReadinessCheck(
                name="database",
                ready=False,
                message=f"postgresql unavailable: {exc}",
                retryable=True,
            )
        return ReadinessCheck(name="database", ready=True, message="connected")

    async def _check_minio(self) -> ReadinessCheck:
        if self._minio_store is None:
            return ReadinessCheck(
                name="minio",
                ready=False,
                message="store not configured",
                retryable=True,
            )
        try:
            await self._minio_store.check_access()
        except Exception as exc:
            return ReadinessCheck(
                name="minio",
                ready=False,
                message=f"minio unavailable: {exc}",
                retryable=True,
            )
        return ReadinessCheck(name="minio", ready=True, message="bucket accessible")

    async def _check_qdrant(self) -> ReadinessCheck:
        if self._qdrant_store is None:
            return ReadinessCheck(
                name="qdrant",
                ready=False,
                message="vector store not configured",
                retryable=True,
            )
        try:
            await self._qdrant_store.ensure_collection()
        except Exception as exc:
            return ReadinessCheck(
                name="qdrant",
                ready=False,
                message=f"qdrant unavailable: {exc}",
                retryable=True,
            )
        return ReadinessCheck(name="qdrant", ready=True, message="collection contract ok")

    async def _check_native_module(self) -> ReadinessCheck:
        try:
            import image_runtime  # noqa: F401
        except ModuleNotFoundError:
            return ReadinessCheck(
                name="native_module",
                ready=False,
                message="image_runtime extension not installed",
                retryable=False,
            )
        except Exception as exc:
            return ReadinessCheck(
                name="native_module",
                ready=False,
                message=f"native module import failed: {exc}",
                retryable=True,
            )
        return ReadinessCheck(name="native_module", ready=True, message="extension importable")
