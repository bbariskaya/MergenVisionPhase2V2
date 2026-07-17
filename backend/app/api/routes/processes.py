"""Process query routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.controllers.face_controller import FaceController
from app.api.routes.dependencies import get_face_controller
from app.api.schemas import ProcessResponse
from app.domain.errors import DomainError, ValidationError

router = APIRouter(prefix="/processes", tags=["processes"])


def _error_response(request: Request, code: str, message: str, status_code: int) -> HTTPException:
    request_id = str(getattr(request.state, "request_id", "unknown"))
    return HTTPException(
        status_code=status_code,
        detail={
            "requestId": request_id,
            "error": {
                "code": code,
                "message": message,
                "retryable": False,
                "details": {},
            },
        },
    )


def _handle_domain_error(request: Request, exc: DomainError) -> HTTPException:
    request_id = str(getattr(request.state, "request_id", "unknown"))
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "requestId": request_id,
            "error": {
                "code": "INVALID_REQUEST",
                "message": str(exc),
                "retryable": False,
                "details": {},
            },
        },
    )


@router.get("/{process_id}", response_model=ProcessResponse)
async def get_process(
    request: Request,
    process_id: str,
    controller: FaceController = Depends(get_face_controller),
) -> ProcessResponse:
    try:
        process = await controller.get_process(
            request_id=str(request.state.request_id), process_id_str=process_id
        )
    except ValidationError as exc:
        raise _handle_domain_error(request, exc) from exc

    if process is None:
        raise _error_response(
            request,
            code="PROCESS_NOT_FOUND",
            message=f"Process {process_id} not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return process
