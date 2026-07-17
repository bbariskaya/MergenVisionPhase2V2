"""Face recognition API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, Request, Response, UploadFile

from app.api.controllers.face_controller import FaceController, RecognizeRequestData
from app.api.routes.common import raise_not_found
from app.api.routes.dependencies import get_face_controller
from app.api.schemas import (
    EnrollByFaceIdRequest,
    EnrollResponse,
    FaceHistoryResponse,
    FaceSampleResponse,
    FaceSamplesResponse,
    IdentityDetailResponse,
    IdentityListResponse,
    RecognizeResponse,
)
from app.application.services.image_validation_service import ImageValidator
from app.infrastructure.config import settings

router = APIRouter(prefix="/faces", tags=["faces"])

_image_validator = ImageValidator(
    max_image_bytes=settings.max_image_bytes,
    max_image_width=settings.max_image_width,
    max_image_height=settings.max_image_height,
    max_image_pixels=settings.max_image_pixels,
)


async def _read_bounded(image: UploadFile, max_bytes: int) -> bytes:
    """Stream-read the uploaded file up to max_bytes + 1."""
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
    return await controller.recognize(
        request_id=str(request.state.request_id),
        data=RecognizeRequestData(image_bytes=image_bytes, filename=filename),
    )


@router.post("/{face_id}/enroll", response_model=EnrollResponse)
async def enroll_face(
    request: Request,
    face_id: str,
    body: EnrollByFaceIdRequest,
    controller: FaceController = Depends(get_face_controller),
) -> EnrollResponse:
    return await controller.enroll(
        request_id=str(request.state.request_id),
        face_id=face_id,
        display_name=body.name,
        metadata=body.metadata or {},
    )


@router.get("/{face_id}", response_model=IdentityDetailResponse)
async def get_identity(
    request: Request,
    face_id: str,
    controller: FaceController = Depends(get_face_controller),
) -> IdentityDetailResponse:
    identity = await controller.get_identity(
        request_id=str(request.state.request_id), face_id_str=face_id
    )
    if identity is None:
        raise_not_found(
            request,
            code="FACE_NOT_FOUND",
            message=f"Face identity {face_id} not found.",
        )
    return identity


@router.delete("/{face_id}", response_model=EnrollResponse)
async def delete_identity(
    request: Request,
    face_id: str,
    controller: FaceController = Depends(get_face_controller),
) -> EnrollResponse:
    return await controller.delete_identity(
        request_id=str(request.state.request_id), face_id_str=face_id
    )


@router.get("", response_model=IdentityListResponse)
async def list_identities(
    request: Request,
    search: str | None = Query(None, description="Optional case-insensitive name filter"),
    controller: FaceController = Depends(get_face_controller),
) -> IdentityListResponse:
    return await controller.list_identities(
        request_id=str(request.state.request_id),
        query=search,
    )


@router.get("/{face_id}/samples", response_model=FaceSamplesResponse)
async def get_face_samples(
    request: Request,
    face_id: str,
    controller: FaceController = Depends(get_face_controller),
) -> FaceSamplesResponse:
    samples = await controller.get_face_samples(
        request_id=str(request.state.request_id), face_id_str=face_id
    )
    if samples is None:
        raise_not_found(
            request,
            code="FACE_NOT_FOUND",
            message=f"Face identity {face_id} not found.",
        )
    return samples


@router.post("/{face_id}/samples", response_model=FaceSampleResponse)
async def add_face_sample(
    request: Request,
    face_id: str,
    image: UploadFile = File(..., description="JPEG image containing the person's face"),
    controller: FaceController = Depends(get_face_controller),
) -> FaceSampleResponse:
    image_bytes = await _read_bounded(image, _image_validator._max_image_bytes)
    _image_validator.validate(image_bytes)
    sample = await controller.add_face_sample(
        request_id=str(request.state.request_id),
        face_id_str=face_id,
        image_bytes=image_bytes,
    )
    if sample is None:
        raise_not_found(
            request,
            code="FACE_NOT_FOUND",
            message=f"Face identity {face_id} not found.",
        )
    return sample


@router.delete("/{face_id}/samples/{sample_id}", status_code=204)
async def delete_face_sample(
    request: Request,
    face_id: str,
    sample_id: str,
    controller: FaceController = Depends(get_face_controller),
) -> Response:
    await controller.delete_face_sample(
        request_id=str(request.state.request_id),
        face_id_str=face_id,
        sample_id_str=sample_id,
    )
    return Response(status_code=204)


@router.get("/{face_id}/samples/{sample_id}/image")
async def get_sample_image(
    request: Request,
    face_id: str,
    sample_id: str,
    controller: FaceController = Depends(get_face_controller),
) -> Response:
    result = await controller.get_sample_image_bytes(face_id_str=face_id, sample_id_str=sample_id)
    if result is None:
        raise_not_found(
            request,
            code="SAMPLE_NOT_FOUND",
            message=f"Sample {sample_id} for face {face_id} not found.",
        )
    data, content_type = result
    return Response(content=data, media_type=content_type)


@router.get("/{face_id}/history", response_model=FaceHistoryResponse)
async def get_face_history(
    request: Request,
    face_id: str,
    controller: FaceController = Depends(get_face_controller),
) -> FaceHistoryResponse:
    return await controller.get_history(
        request_id=str(request.state.request_id), face_id_str=face_id
    )
