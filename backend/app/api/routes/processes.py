"""Process query routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.controllers.face_controller import FaceController
from app.api.routes.dependencies import get_face_controller
from app.api.schemas import ProcessResponse
from app.domain.errors import ValidationError

router = APIRouter(prefix="/processes", tags=["processes"])


@router.get("/{process_id}", response_model=ProcessResponse)
async def get_process(
    process_id: str,
    controller: FaceController = Depends(get_face_controller),
) -> ProcessResponse:
    try:
        process = await controller.get_process(process_id)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": exc.__class__.__name__, "message": str(exc)},
        ) from exc

    if process is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": f"Process {process_id} not found"},
        )
    return process
