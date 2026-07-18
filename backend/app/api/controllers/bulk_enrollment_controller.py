"""HTTP-facing controller for bulk photo enrollment."""

from __future__ import annotations

from app.api.schemas import BulkEnrollRequest, BulkEnrollResponse
from app.application.services.bulk_enrollment_service import (
    BulkEnrollmentService,
    EnrollmentIdentity,
    EnrollmentPhoto,
)


class BulkEnrollmentController:
    def __init__(self, service: BulkEnrollmentService) -> None:
        self._service = service

    async def enroll_batch(
        self,
        request_id: str,
        data: BulkEnrollRequest,
    ) -> BulkEnrollResponse:
        identities = [_to_domain(item) for item in data.identities]
        result = await self._service.enroll_batch(identities)
        return BulkEnrollResponse(
            request_id=request_id,
            process_id=result.process_id,
            discovered_identities=result.discovered_identities,
            discovered_photos=result.discovered_photos,
            enrolled_identities=result.enrolled_identities,
            enrolled_photos=result.enrolled_photos,
            no_face=result.no_face,
            decode_error=result.decode_error,
            failed=result.failed,
            errors=result.errors,
        )


def _to_domain(item: BulkEnrollRequest.identities.__class__) -> EnrollmentIdentity:  # type: ignore[name-defined]
    photos: list[EnrollmentPhoto] = []
    for photo in item.photos:
        enrollment_photo = BulkEnrollmentService.photo_from_base64(
            photo.filename,
            photo.image_base64,
        )
        photos.append(enrollment_photo)
    return EnrollmentIdentity(
        display_name=item.display_name,
        photos=photos,
        metadata=item.metadata or {},
        source_dataset=item.source_dataset,
    )
