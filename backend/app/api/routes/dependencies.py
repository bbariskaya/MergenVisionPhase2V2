"""FastAPI dependency injection wiring.

Dependencies are exposed as callables so they can be overridden in tests.
"""

from __future__ import annotations

from typing import cast

from fastapi import HTTPException, Request, status

from app.api.controllers.face_controller import FaceController
from app.application.services.video_overlay_service import VideoOverlayService
from app.application.services.video_result_service import VideoResultService
from app.application.services.video_upload_service import VideoUploadService
from app.infrastructure.config import settings
from app.infrastructure.persistence.sqlalchemy.session import async_session_maker
from app.infrastructure.persistence.sqlalchemy.unit_of_work import SqlAlchemyUnitOfWork
from app.infrastructure.storage.minio_adapter import MinIOObjectStore


def get_face_controller(request: Request) -> FaceController:
    """Return the singleton face controller from app state.

    Raises HTTP 503 if the application is not ready (e.g. native runtime or
    storage dependencies are unavailable).
    """
    controller = getattr(request.app.state, "face_controller", None)
    if controller is None:
        request_id = str(getattr(request.state, "request_id", "unknown"))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "requestId": request_id,
                "error": {
                    "code": "DEPENDENCY_UNAVAILABLE",
                    "message": "Service is not ready. Check /health/ready for details.",
                    "retryable": True,
                    "details": {},
                },
            },
        )
    return cast(FaceController, controller)


def get_video_upload_service(request: Request) -> VideoUploadService:
    """Return the video upload service from app state."""
    service = getattr(request.app.state, "video_upload_service", None)
    if service is None:
        request_id = str(getattr(request.state, "request_id", "unknown"))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "requestId": request_id,
                "error": {
                    "code": "DEPENDENCY_UNAVAILABLE",
                    "message": "Service is not ready. Check /health/ready for details.",
                    "retryable": True,
                    "details": {},
                },
            },
        )
    return cast(VideoUploadService, service)


def get_video_result_service() -> VideoResultService:
    """Return the video result read service."""
    unit_of_work_factory = lambda: SqlAlchemyUnitOfWork(async_session_maker)  # noqa: E731
    return VideoResultService(unit_of_work_factory=unit_of_work_factory)


def get_object_store() -> MinIOObjectStore:
    """Return a fresh MinIO object store adapter."""
    return MinIOObjectStore()


def get_video_overlay_service() -> VideoOverlayService:
    """Return the video overlay read/write service."""
    unit_of_work_factory = lambda: SqlAlchemyUnitOfWork(async_session_maker)  # noqa: E731
    return VideoOverlayService(
        unit_of_work_factory=unit_of_work_factory,
        object_store=MinIOObjectStore(),
        bucket_name=settings.minio_bucket_name,
    )
