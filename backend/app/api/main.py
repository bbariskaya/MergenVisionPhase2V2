"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.controllers.face_controller import FaceController
from app.api.routes import faces, processes
from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.application.services.image_recognition_service import ImageRecognitionService
from app.domain.errors import DomainError
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from app.infrastructure.config import settings
from app.infrastructure.runtime.native_image_recognition_adapter import (
    NativeImageRecognitionAdapter,
)
from app.infrastructure.storage.minio_adapter import MinIOObjectStore
from app.infrastructure.uuid7 import Uuid7Generator
from app.infrastructure.vectors.qdrant_adapter import QdrantVectorStore


def _create_image_recognition_service() -> ImageRecognitionService:
    unit_of_work_factory = lambda: SqlAlchemyUnitOfWork(async_session_maker)  # noqa: E731
    object_store = MinIOObjectStore()
    vector_store = QdrantVectorStore()
    id_generator = Uuid7Generator()

    lifecycle_service = IdentityStorageLifecycleService(
        unit_of_work_factory=unit_of_work_factory,
        object_store=object_store,
        vector_store=vector_store,
        id_generator=id_generator,
    )

    return ImageRecognitionService(
        lifecycle_service=lifecycle_service,
        unit_of_work_factory=unit_of_work_factory,
        max_image_bytes=settings.max_image_bytes,
        model_version=settings.model_version,
        engine_factory=NativeImageRecognitionAdapter,
        match_threshold=settings.match_threshold,
    )


def _create_face_controller() -> FaceController:
    service = _create_image_recognition_service()
    return FaceController(service)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="MergenVision",
        description="GPU-accelerated persistent face identity API",
        version="0.2.0",
        lifespan=_lifespan,
    )

    app.state.face_controller = _create_face_controller()

    @app.exception_handler(DomainError)
    async def _domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": exc.__class__.__name__, "message": str(exc)},
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(faces.router)
    app.include_router(processes.router)
    return app
