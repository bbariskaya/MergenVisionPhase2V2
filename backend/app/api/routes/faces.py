"""Face recognition API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.api.controllers.face_controller import FaceController, RecognizeRequestData
from app.api.routes.dependencies import get_face_controller
from app.api.schemas import (
    EnrollByFaceIdRequest,
    EnrollResponse,
    FaceHistoryResponse,
    IdentityDetailResponse,
    RecognizeResponse,
)
from app.application.services.image_validation_service import ImageValidator
from app.domain.errors import DomainError, ValidationError
from app.infrastructure.config import settings

router = APIRouter(prefix="/faces", tags=["faces"])

_image_validator = ImageValidator(
    max_image_bytes=settings.max_image_bytes,
    max_image_width=settings.max_image_width,
    max_image_height=settings.max_image_height,
    max_image_pixels=settings.max_image_pixels,
)


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
    code = "INVALID_REQUEST"
    if isinstance(exc, ValidationError):
        code = "INVALID_REQUEST"
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "requestId": request_id,
            "error": {
                "code": code,
                "message": str(exc),
                "retryable": False,
                "details": {},
            },
        },
    )


def _to_face_response_schema(controller_response: RecognizeResponse) -> RecognizeResponse:
    # The controller already builds the schema-ready object.
    return controller_response


async def _read_bounded(image: UploadFile, max_bytes: int) -> bytes:
    """Stream-read the uploaded file up to max_bytes + 1.

    Raises PayloadTooLargeError if the stream would exceed the limit.
    """
    from app.domain.errors import PayloadTooLargeError

    data = bytearray()
    remaining = max_bytes + 1
    while remaining > 0:
        chunk = await image.read(remaining)
        if not chunk:
            break
        data.extend(chunk)
        remaining -= len(chunk)
    if remaining == 0:
        raise PayloadTooLargeError(
            f"image exceeds maximum allowed size of {max_bytes} bytes"
        )
    return bytes(data)


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize_faces(
    request: Request,
    image: UploadFile = File(..., description="JPEG image containing one or more faces"),
    controller: FaceController = Depends(get_face_controller),
) -> RecognizeResponse:
    filename = image.filename or "upload.jpg"
    image_bytes = await _read_bounded(image, _image_validator._max_image_bytes)
    _image_validator.validate(image_bytes)
    try:
        return await controller.recognize(
            request_id=str(request.state.request_id),
            data=RecognizeRequestData(image_bytes=image_bytes, filename=filename),
        )
    except ValidationError as exc:
        raise _handle_domain_error(request, exc) from exc
    except DomainError as exc:
        raise _handle_domain_error(request, exc) from exc


@router.post("/{face_id}/enroll", response_model=EnrollResponse)
async def enroll_face(
    request: Request,
    face_id: str,
    body: EnrollByFaceIdRequest,
    controller: FaceController = Depends(get_face_controller),
) -> EnrollResponse:
    try:
        return await controller.enroll(
            request_id=str(request.state.request_id),
            face_id=face_id,
            display_name=body.name,
            metadata=body.metadata or {},
        )
    except ValidationError as exc:
        raise _handle_domain_error(request, exc) from exc
    except DomainError as exc:
        raise _handle_domain_error(request, exc) from exc


@router.get("/{face_id}", response_model=IdentityDetailResponse)
async def get_identity(
    request: Request,
    face_id: str,
    controller: FaceController = Depends(get_face_controller),
) -> IdentityDetailResponse:
    try:
        identity = await controller.get_identity(request_id=str(request.state.request_id), face_id_str=face_id)
    except ValidationError as exc:
        raise _handle_domain_error(request, exc) from exc

    if identity is None:
        raise _error_response(
            request,
            code="FACE_NOT_FOUND",
            message=f"Face identity {face_id} not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return identity


@router.delete("/{face_id}", response_model=EnrollResponse)
async def delete_identity(
    request: Request,
    face_id: str,
    controller: FaceController = Depends(get_face_controller),
) -> EnrollResponse:
    try:
        return await controller.delete_identity(
            request_id=str(request.state.request_id), face_id_str=face_id
        )
    except ValidationError as exc:
        raise _handle_domain_error(request, exc) from exc
    except DomainError as exc:
        raise _handle_domain_error(request, exc) from exc


@router.get("/{face_id}/history", response_model=FaceHistoryResponse)
async def get_face_history(
    request: Request,
    face_id: str,
    controller: FaceController = Depends(get_face_controller),
) -> FaceHistoryResponse:
    try:
        return await controller.get_history(
            request_id=str(request.state.request_id), face_id_str=face_id
        )
    except ValidationError as exc:
        raise _handle_domain_error(request, exc) from exc
    except DomainError as exc:
        raise _handle_domain_error(request, exc) from exc
