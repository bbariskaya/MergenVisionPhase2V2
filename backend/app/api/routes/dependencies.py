"""FastAPI dependency injection wiring.

Dependencies are exposed as callables so they can be overridden in tests.
"""

from __future__ import annotations

from typing import cast

from fastapi import HTTPException, Request, status

from app.api.controllers.face_controller import FaceController
from app.application.services.video_upload_service import VideoUploadService


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
