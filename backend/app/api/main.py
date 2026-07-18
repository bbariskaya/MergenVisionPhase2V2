"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from fastapi import FastAPI, Request, status
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse, Response

from app.api.controllers.bulk_enrollment_controller import BulkEnrollmentController
from app.api.controllers.face_controller import FaceController
from app.api.controllers.person_controller import PersonController
from app.api.routes import faces, people, processes, videos
from app.application.services.bulk_enrollment_service import BulkEnrollmentService
from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.application.services.image_recognition_service import ImageRecognitionService
from app.application.services.person_management_service import PersonManagementService
from app.application.services.video_upload_service import VideoUploadService
from app.domain.errors import (
    DomainError,
    IdempotencyConflictError,
    InvalidMediaError,
    JobNotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
    VideoNotFoundError,
)
from app.infrastructure.config import settings
from app.infrastructure.health.readiness import (
    DefaultReadinessProbe,
    ReadinessCheck,
    ReadinessProbe,
    ReadinessReport,
)
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from app.infrastructure.runtime.native_image_recognition_adapter import (
    NativeImageRecognitionAdapter,
)
from app.infrastructure.storage.minio_adapter import MinIOObjectStore
from app.infrastructure.uuid7 import Uuid7Generator
from app.infrastructure.vectors.qdrant_adapter import QdrantVectorStore


def _generate_request_id() -> str:
    return str(UUID(int=Uuid7Generator().new_uuid7().int))


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
    object_store = MinIOObjectStore()
    return FaceController(service, object_store=object_store)


def _create_person_controller() -> PersonController:
    unit_of_work_factory = lambda: SqlAlchemyUnitOfWork(async_session_maker)  # noqa: E731
    service = PersonManagementService(unit_of_work_factory=unit_of_work_factory)
    return PersonController(service)


def _create_bulk_enrollment_controller() -> BulkEnrollmentController:
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
    person_service = PersonManagementService(unit_of_work_factory=unit_of_work_factory)

    service = BulkEnrollmentService(
        lifecycle_service=lifecycle_service,
        person_service=person_service,
        engine_factory=NativeImageRecognitionAdapter,
        model_version=settings.model_version,
        max_image_bytes=settings.max_image_bytes,
    )
    return BulkEnrollmentController(service)


def _create_video_upload_service() -> VideoUploadService:
    unit_of_work_factory = lambda: SqlAlchemyUnitOfWork(async_session_maker)  # noqa: E731
    object_store = MinIOObjectStore()
    id_generator = Uuid7Generator()

    return VideoUploadService(
        unit_of_work_factory=unit_of_work_factory,
        object_store=object_store,
        id_generator=id_generator,
        bucket_name=settings.minio_bucket_name,
        ffprobe_command=settings.ffprobe_command,
        max_video_bytes=settings.max_video_bytes,
        max_duration_ns=settings.max_video_duration_ns,
        max_display_width=settings.max_video_display_width,
        max_display_height=settings.max_video_display_height,
        allowed_containers=set(settings.allowed_video_containers),
        allowed_codecs=set(settings.allowed_video_codecs),
        retention_seconds=settings.video_retention_seconds,
        staging_prefix=settings.video_staging_prefix,
        source_prefix=settings.video_source_prefix,
        temp_dir=settings.video_temp_dir,
        probe_timeout_seconds=settings.video_probe_timeout_seconds,
        max_attempts=settings.video_max_attempts,
    )


def _safe_error_body(request_id: str, code: str, message: str, retryable: bool = False) -> dict[str, Any]:
    return {
        "requestId": request_id,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "details": {},
        },
    }


def create_app(readiness_probe: ReadinessProbe | None = None) -> FastAPI:
    if readiness_probe is None:
        readiness_probe = DefaultReadinessProbe(
            settings_obj=settings,
            session_maker=async_session_maker,
            minio_store=MinIOObjectStore(),
            qdrant_store=QdrantVectorStore(),
        )

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        logger = logging.getLogger(__name__)
        report = await readiness_probe.check()
        app.state.readiness_report = report

        try:
            app.state.video_upload_service = _create_video_upload_service()
        except Exception as exc:
            logger.exception("video_upload_service initialization failed: %s", exc)
            app.state.video_upload_service = None

        if report.ready:
            app.state.face_controller = None
            app.state.person_controller = None
            app.state.bulk_enrollment_controller = None

            controller_errors: list[str] = []
            try:
                app.state.face_controller = _create_face_controller()
            except Exception as exc:
                controller_errors.append(f"face_controller: {exc}")
                logger.exception("face_controller initialization failed: %s", exc)

            try:
                app.state.person_controller = _create_person_controller()
            except Exception as exc:
                controller_errors.append(f"person_controller: {exc}")
                logger.exception("person_controller initialization failed: %s", exc)

            try:
                app.state.bulk_enrollment_controller = _create_bulk_enrollment_controller()
            except Exception as exc:
                controller_errors.append(f"bulk_enrollment_controller: {exc}")
                logger.exception("bulk_enrollment_controller initialization failed: %s", exc)

            if controller_errors:
                controller_report = ReadinessReport(
                    ready=False,
                    status="not_ready",
                    message=f"controller_init_failed: {'; '.join(controller_errors)}",
                    checks=report.checks + (
                        ReadinessCheck(
                            name="controllers",
                            ready=False,
                            message="one or more controllers failed to initialize",
                            retryable=True,
                        ),
                    ),
                )
                app.state.readiness_report = controller_report
        else:
            app.state.face_controller = None
            app.state.person_controller = None
            app.state.bulk_enrollment_controller = None

        yield

        app.state.face_controller = None
        app.state.person_controller = None
        app.state.bulk_enrollment_controller = None
        app.state.video_upload_service = None

    app = FastAPI(
        title="MergenVision",
        description="GPU-accelerated persistent face identity API",
        version="0.2.0",
        lifespan=_lifespan,
    )

    @app.middleware("http")
    async def _request_id_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = request.headers.get("X-Request-ID") or _generate_request_id()
        request.state.request_id = request_id
        response = await call_next(request)
        if not response.headers.get("X-Request-ID"):
            response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(DomainError)
    async def _domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", "unknown"))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            headers={"X-Request-ID": request_id},
            content=_safe_error_body(request_id, "INVALID_REQUEST", str(exc)),
        )

    @app.exception_handler(PayloadTooLargeError)
    async def _payload_too_large_handler(request: Request, exc: PayloadTooLargeError) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", "unknown"))
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            headers={"X-Request-ID": request_id},
            content=_safe_error_body(request_id, "PAYLOAD_TOO_LARGE", str(exc)),
        )

    @app.exception_handler(UnsupportedMediaTypeError)
    async def _unsupported_media_handler(
        request: Request, exc: UnsupportedMediaTypeError
    ) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", "unknown"))
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            headers={"X-Request-ID": request_id},
            content=_safe_error_body(request_id, "UNSUPPORTED_MEDIA_TYPE", str(exc)),
        )

    @app.exception_handler(InvalidMediaError)
    async def _invalid_media_handler(request: Request, exc: InvalidMediaError) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", "unknown"))
        message = str(exc).lower()
        if "probe timed out" in message or "probe timeout" in message:
            code = "VIDEO_PROBE_TIMEOUT"
        elif "ffprobe" in message:
            code = "VIDEO_PROBE_FAILED"
        else:
            code = "INVALID_MEDIA"
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            headers={"X-Request-ID": request_id},
            content=_safe_error_body(
                request_id,
                code,
                "Video could not be probed or is invalid."
                if code.startswith("VIDEO_PROBE")
                else str(exc),
            ),
        )

    @app.exception_handler(IdempotencyConflictError)
    async def _idempotency_conflict_handler(
        request: Request, exc: IdempotencyConflictError
    ) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", "unknown"))
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            headers={"X-Request-ID": request_id},
            content=_safe_error_body(request_id, "IDEMPOTENCY_CONFLICT", str(exc)),
        )

    @app.exception_handler(JobNotFoundError)
    async def _job_not_found_handler(request: Request, exc: JobNotFoundError) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", "unknown"))
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            headers={"X-Request-ID": request_id},
            content=_safe_error_body(request_id, "JOB_NOT_FOUND", str(exc)),
        )

    @app.exception_handler(VideoNotFoundError)
    async def _video_not_found_handler(
        request: Request, exc: VideoNotFoundError
    ) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", "unknown"))
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            headers={"X-Request-ID": request_id},
            content=_safe_error_body(request_id, "VIDEO_NOT_FOUND", str(exc)),
        )

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", "unknown"))
        detail = exc.detail if isinstance(exc.detail, dict) else _safe_error_body(
            request_id,
            "INTERNAL_ERROR" if exc.status_code >= 500 else "INVALID_REQUEST",
            str(exc.detail),
        )
        if isinstance(detail, dict) and "requestId" not in detail:
            detail["requestId"] = request_id
        return JSONResponse(
            status_code=exc.status_code,
            headers={"X-Request-ID": request_id},
            content=detail,
        )

    @app.exception_handler(Exception)
    async def _generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", "unknown"))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            headers={"X-Request-ID": request_id},
            content=_safe_error_body(
                request_id,
                "INTERNAL_ERROR",
                "An unexpected internal error occurred.",
            ),
        )

    @app.get("/health/live")
    async def health_live(request: Request) -> dict[str, str]:
        return {"status": "ok", "requestId": getattr(request.state, "request_id", "unknown")}

    @app.get("/health/ready")
    async def health_ready(request: Request) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", "unknown"))
        report: ReadinessReport | None = getattr(app.state, "readiness_report", None)
        if report is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                headers={"X-Request-ID": request_id},
                content={
                    "requestId": request_id,
                    "ready": False,
                    "status": "not_ready",
                    "message": "readiness not initialized",
                    "checks": [],
                },
            )

        body = {
            "requestId": request_id,
            "ready": report.ready,
            "status": report.status,
            "message": report.message,
            "checks": [
                {"name": c.name, "ready": c.ready, "message": c.message, "retryable": c.retryable}
                for c in report.checks
            ],
        }
        if report.ready:
            return JSONResponse(headers={"X-Request-ID": request_id}, content=body)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"X-Request-ID": request_id},
            content=body,
        )

    app.include_router(faces.router, prefix="/api/v1")
    app.include_router(people.router, prefix="/api/v1")
    app.include_router(processes.router, prefix="/api/v1")
    app.include_router(videos.router, prefix="/api/v1")
    return app
