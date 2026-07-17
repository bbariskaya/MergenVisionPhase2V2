"""Process query routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.controllers.face_controller import FaceController
from app.api.routes.common import raise_not_found
from app.api.routes.dependencies import get_face_controller
from app.api.schemas import ProcessResponse

router = APIRouter(prefix="/processes", tags=["processes"])


@router.get("/{process_id}", response_model=ProcessResponse)
async def get_process(
    request: Request,
    process_id: str,
    controller: FaceController = Depends(get_face_controller),
) -> ProcessResponse:
    process = await controller.get_process(
        request_id=str(request.state.request_id), process_id_str=process_id
    )
    if process is None:
        raise_not_found(
            request,
            code="PROCESS_NOT_FOUND",
            message=f"Process {process_id} not found.",
        )
    return process
