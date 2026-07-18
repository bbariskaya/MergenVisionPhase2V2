"""Photo-based bulk enrollment service.

This is the backend-logic scaffold for Milestone B. It enrolls many people from
a manifest of photos using the existing single-image inference engine. The GPU
hot-path batching demonstrated in MergenVisionDemo will be wired in later by
extending ``ImageRecognitionEngine`` with a batch method and the native runtime
with ``infer_jpeg_batch``.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.application.ports.image_recognition import (
    ImageRecognitionEngine,
    NativeFaceDetection,
)
from app.application.services.identity_storage_lifecycle_service import (
    IdentityStorageLifecycleService,
)
from app.application.services.person_management_service import PersonManagementService
from app.domain.entities.person import Person
from app.domain.entities.process_record import ProcessRecord
from app.domain.errors import ValidationError
from app.domain.value_objects import FaceId

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrollmentPhoto:
    filename: str
    data: bytes
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.data:
            raise ValidationError("photo data is empty")


@dataclass(frozen=True)
class EnrollmentIdentity:
    display_name: str
    photos: list[EnrollmentPhoto]
    metadata: dict[str, Any] = field(default_factory=dict)
    source_dataset: str | None = None

    def __post_init__(self) -> None:
        if not self.display_name or not self.display_name.strip():
            raise ValidationError("display_name is required")
        if not self.photos:
            raise ValidationError("at least one photo is required")


@dataclass
class BulkEnrollmentResult:
    process_id: str
    discovered_identities: int = 0
    discovered_photos: int = 0
    enrolled_identities: int = 0
    enrolled_photos: int = 0
    no_face: int = 0
    decode_error: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


class BulkEnrollmentService:
    """Enroll a batch of people from photos deterministically and with compensation."""

    def __init__(
        self,
        lifecycle_service: IdentityStorageLifecycleService,
        person_service: PersonManagementService,
        engine_factory: Callable[[], ImageRecognitionEngine],
        *,
        model_version: str,
        max_image_bytes: int = 26_214_400,
        max_batch_size: int = 1000,
    ) -> None:
        self._lifecycle = lifecycle_service
        self._person_service = person_service
        self._engine_factory = engine_factory
        self._engine: ImageRecognitionEngine | None = None
        self._model_version = model_version
        self._max_image_bytes = max_image_bytes
        self._max_batch_size = max_batch_size

    def _get_engine(self) -> ImageRecognitionEngine:
        if self._engine is None:
            self._engine = self._engine_factory()
        return self._engine

    async def enroll_batch(
        self,
        identities: list[EnrollmentIdentity],
    ) -> BulkEnrollmentResult:
        if not identities:
            raise ValidationError("batch cannot be empty")
        if len(identities) > self._max_batch_size:
            raise ValidationError(f"batch size cannot exceed {self._max_batch_size}")

        process = await self._lifecycle.start_process(
            "bulk_enroll",
            {"model_version": self._model_version, "identity_count": len(identities)},
        )
        result = BulkEnrollmentResult(process_id=str(process.process_id))

        try:
            for identity in identities:
                await self._enroll_identity(process, identity, result)
        except Exception as exc:
            result.errors.append(str(exc))
            logger.exception("Bulk enrollment shard failed")
            await self._lifecycle.fail_process(
                process.process_id,
                "bulk_enrollment_failed",
                {"error": str(exc)},
            )
            raise
        finally:
            if process.status == "processing":
                await self._lifecycle.complete_process(
                    process.process_id,
                    face_count=result.enrolled_photos,
                    details={
                        "discovered_identities": result.discovered_identities,
                        "discovered_photos": result.discovered_photos,
                        "enrolled_identities": result.enrolled_identities,
                        "enrolled_photos": result.enrolled_photos,
                        "no_face": result.no_face,
                        "decode_error": result.decode_error,
                        "failed": result.failed,
                    },
                )

        return result

    async def _enroll_identity(
        self,
        process: ProcessRecord,
        identity: EnrollmentIdentity,
        result: BulkEnrollmentResult,
    ) -> None:
        result.discovered_identities += 1
        result.discovered_photos += len(identity.photos)

        person = await self._person_service.create_person(
            display_name=identity.display_name,
            metadata={
                **identity.metadata,
                "source_dataset": identity.source_dataset,
            },
        )

        canonical_face_id: FaceId | None = None
        for photo in identity.photos:
            try:
                outcome = await self._enroll_photo(
                    process=process,
                    person=person,
                    photo=photo,
                    canonical_face_id=canonical_face_id,
                    display_name=identity.display_name,
                    metadata=identity.metadata,
                )
            except _DecodeError:
                result.decode_error += 1
                continue
            except _NoFaceError:
                result.no_face += 1
                continue
            except Exception as exc:
                result.failed += 1
                result.errors.append(f"{identity.display_name}/{photo.filename}: {exc}")
                logger.warning("Photo enrollment failed: %s", exc)
                continue

            result.enrolled_photos += 1
            if canonical_face_id is None:
                canonical_face_id = outcome.face_id

        if canonical_face_id is not None:
            result.enrolled_identities += 1

    async def _enroll_photo(
        self,
        process: ProcessRecord,
        person: Person,
        photo: EnrollmentPhoto,
        canonical_face_id: FaceId | None,
        display_name: str,
        metadata: dict[str, Any],
    ) -> Any:
        if len(photo.data) > self._max_image_bytes:
            raise _DecodeError(f"image exceeds {self._max_image_bytes} bytes")

        try:
            inference = await self._get_engine().detect_and_embed(photo.data)
        except Exception as exc:
            raise _DecodeError(f"image decode/inference failed: {exc}") from exc

        detection = self._pick_largest_face(inference.detections)
        if detection is None:
            raise _NoFaceError(f"no face detected in {photo.filename}")

        if canonical_face_id is None:
            return await self._lifecycle.create_known_identity(
                process_id=process.process_id,
                person_id=person.person_id,
                display_name=display_name,
                metadata=metadata,
                crop_bytes=detection.aligned_crop_bytes,
                embedding=detection.embedding,
                bbox=detection.bounding_box,
            )

        return await self._lifecycle.add_sample(
            face_id=canonical_face_id,
            crop_bytes=detection.aligned_crop_bytes,
            embedding=detection.embedding,
        )

    @staticmethod
    def _pick_largest_face(detections: list[NativeFaceDetection]) -> NativeFaceDetection | None:
        if not detections:
            return None
        return max(
            detections,
            key=lambda d: d.bounding_box.width * d.bounding_box.height,
        )

    @staticmethod
    def photo_from_base64(filename: str, b64_data: str) -> EnrollmentPhoto:
        try:
            data = base64.b64decode(b64_data)
        except Exception as exc:
            raise ValidationError(f"invalid base64 for {filename}: {exc}") from exc
        return EnrollmentPhoto(
            filename=filename,
            data=data,
            content_sha256=hashlib.sha256(data).hexdigest(),
        )


class _DecodeError(Exception):
    pass


class _NoFaceError(Exception):
    pass
