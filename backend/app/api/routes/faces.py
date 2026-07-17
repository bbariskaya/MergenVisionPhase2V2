"""Face recognition API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status

from app.api.controllers.face_controller import FaceController, RecognizeRequestData
from app.api.routes.dependencies import get_face_controller
from app.api.schemas import (
    EnrollRequest,
    EnrollResponse,
    FaceHistoryResponse,
    IdentityDetailResponse,
    RecognizeResponse,
)
from app.domain.errors import DomainError, ValidationError

router = APIRouter(prefix="/faces", tags=["faces"])


def _handle_domain_error(exc: DomainError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": exc.__class__.__name__, "message": str(exc)},
    )


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize_faces(
    image: UploadFile = File(..., description="JPEG/PNG/WebP image containing one or more faces"),
    controller: FaceController = Depends(get_face_controller),
) -> RecognizeResponse:
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"error": "unsupported_media_type", "message": "File must be an image"},
        )
    try:
        image_bytes = await image.read()
        return await controller.recognize(
            RecognizeRequestData(image_bytes=image_bytes, filename=image.filename)
        )
    except ValidationError as exc:
        raise _handle_domain_error(exc) from exc
    except DomainError as exc:
        raise _handle_domain_error(exc) from exc


@router.post("/enroll", response_model=EnrollResponse)
async def enroll_face(
    request: EnrollRequest,
    controller: FaceController = Depends(get_face_controller),
) -> EnrollResponse:
    try:
        return await controller.enroll(request)
    except ValidationError as exc:
        raise _handle_domain_error(exc) from exc
    except DomainError as exc:
        raise _handle_domain_error(exc) from exc


@router.get("/{face_id}", response_model=IdentityDetailResponse)
async def get_identity(
    face_id: str,
    controller: FaceController = Depends(get_face_controller),
) -> IdentityDetailResponse:
    try:
        identity = await controller.get_identity(face_id)
    except ValidationError as exc:
        raise _handle_domain_error(exc) from exc

    if identity is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": f"Face identity {face_id} not found"},
        )
    return identity


@router.delete("/{face_id}", response_model=None)
async def delete_identity(
    face_id: str,
    controller: FaceController = Depends(get_face_controller),
) -> Response:
    try:
        await controller.delete_identity(face_id)
    except ValidationError as exc:
        raise _handle_domain_error(exc) from exc
    except DomainError as exc:
        raise _handle_domain_error(exc) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{face_id}/history", response_model=FaceHistoryResponse)
async def get_face_history(
    face_id: str,
    controller: FaceController = Depends(get_face_controller),
) -> FaceHistoryResponse:
    try:
        return await controller.get_history(face_id)
    except ValidationError as exc:
        raise _handle_domain_error(exc) from exc
