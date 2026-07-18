"""Person directory API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from app.api.controllers.bulk_enrollment_controller import BulkEnrollmentController
from app.api.controllers.person_controller import (
    CreatePersonRequestData,
    PersonController,
    UpdatePersonRequestData,
)
from app.api.routes.common import raise_not_found
from app.api.routes.dependencies import (
    get_bulk_enrollment_controller,
    get_person_controller,
)
from app.api.schemas import (
    BulkEnrollRequest,
    BulkEnrollResponse,
    EmptyResponse,
    PeopleBatchCreateRequest,
    PeopleBatchCreateResponse,
    PersonDetailResponse,
    PersonListResponse,
    PersonSummary,
)

router = APIRouter(prefix="/people", tags=["people"])


@router.get("", response_model=PersonListResponse)
async def list_people(
    request: Request,
    search: str | None = Query(None, description="Optional case-insensitive name filter"),
    controller: PersonController = Depends(get_person_controller),
) -> PersonListResponse:
    return await controller.list_people(
        request_id=str(request.state.request_id),
        query=search,
    )


@router.post("", response_model=PersonSummary, status_code=201)
async def create_person(
    request: Request,
    body: CreatePersonRequestData,
    controller: PersonController = Depends(get_person_controller),
) -> PersonSummary:
    return await controller.create_person(
        request_id=str(request.state.request_id),
        data=body,
    )


@router.post("/batch", response_model=PeopleBatchCreateResponse, status_code=201)
async def create_people_batch(
    request: Request,
    body: PeopleBatchCreateRequest,
    controller: PersonController = Depends(get_person_controller),
) -> PeopleBatchCreateResponse:
    return await controller.create_people_batch(
        request_id=str(request.state.request_id),
        data=body,
    )


@router.post("/batch-enroll", response_model=BulkEnrollResponse, status_code=201)
async def create_people_batch_enroll(
    request: Request,
    body: BulkEnrollRequest,
    controller: BulkEnrollmentController = Depends(get_bulk_enrollment_controller),
) -> BulkEnrollResponse:
    return await controller.enroll_batch(
        request_id=str(request.state.request_id),
        data=body,
    )


@router.get("/{person_id}", response_model=PersonDetailResponse)
async def get_person(
    request: Request,
    person_id: str,
    controller: PersonController = Depends(get_person_controller),
) -> PersonDetailResponse:
    detail = await controller.get_person(
        request_id=str(request.state.request_id),
        person_id_str=person_id,
    )
    if detail is None:
        raise_not_found(
            request,
            code="PERSON_NOT_FOUND",
            message=f"Person {person_id} not found.",
        )
    return detail


@router.patch("/{person_id}", response_model=PersonSummary)
async def update_person(
    request: Request,
    person_id: str,
    body: UpdatePersonRequestData,
    controller: PersonController = Depends(get_person_controller),
) -> PersonSummary:
    return await controller.update_person(
        request_id=str(request.state.request_id),
        person_id_str=person_id,
        data=body,
    )


@router.delete("/{person_id}", response_model=EmptyResponse, status_code=200)
async def delete_person(
    request: Request,
    person_id: str,
    controller: PersonController = Depends(get_person_controller),
) -> EmptyResponse:
    await controller.deactivate_person(
        request_id=str(request.state.request_id),
        person_id_str=person_id,
    )
    return EmptyResponse(request_id=str(request.state.request_id))
