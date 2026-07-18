"""Request and response contracts for the MergenVision API.

All public response fields use camelCase aliases. Internal Python code keeps
snake_case names. Use `populate_by_name=True` so model constructors accept both
forms during tests and service code.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _PublicBaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class ErrorDetail(_PublicBaseModel):
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(_PublicBaseModel):
    request_id: str
    error: ErrorDetail


class BoundingBoxSchema(_PublicBaseModel):
    x: int
    y: int
    width: int
    height: int


class FaceResponse(_PublicBaseModel):
    face_id: str
    status: str = Field(..., pattern=r"^(known|anonymous|new_anonymous)$")
    name: str | None = None
    metadata: dict[str, Any] | None = None
    bounding_box: BoundingBoxSchema
    confidence: float = Field(..., ge=0.0, le=1.0)


class RecognizeRequest(_PublicBaseModel):
    """Placeholder for any future JSON body on the recognize endpoint."""


class RecognizeResponse(_PublicBaseModel):
    request_id: str
    process_id: str
    status: str
    face_count: int
    faces: list[FaceResponse]


class EnrollByFaceIdRequest(_PublicBaseModel):
    name: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = None


class EnrollRequest(_PublicBaseModel):
    face_id: str
    name: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = None


class EnrollExistingPersonRequest(_PublicBaseModel):
    person_id: str


class AssignFaceToPersonRequest(_PublicBaseModel):
    person_id: str


class EnrollResponse(_PublicBaseModel):
    request_id: str
    process_id: str
    face_id: str
    status: str
    name: str
    metadata: dict[str, Any]
    person_id: str | None = None


class IdentityDetailResponse(_PublicBaseModel):
    request_id: str
    face_id: str
    status: str
    name: str | None = None
    person_id: str | None = None
    metadata: dict[str, Any]
    created_at: str | None = None
    updated_at: str | None = None


class HistoryEntry(_PublicBaseModel):
    process_id: str
    timestamp: str
    process_type: str | None = None
    status: str | None = None
    recognition_status: str | None = None
    match_confidence: float | None = None


class FaceHistoryResponse(_PublicBaseModel):
    request_id: str
    face_id: str
    history: list[HistoryEntry]


class IdentitySummary(_PublicBaseModel):
    face_id: str
    status: str
    name: str | None = None
    metadata: dict[str, Any]
    created_at: str | None = None
    updated_at: str | None = None


class IdentityListResponse(_PublicBaseModel):
    request_id: str
    count: int
    identities: list[IdentitySummary]


class PersonSummary(_PublicBaseModel):
    person_id: str
    display_name: str
    is_active: bool
    face_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None


class CreatePersonItem(_PublicBaseModel):
    display_name: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = None


class PeopleBatchCreateRequest(_PublicBaseModel):
    people: list[CreatePersonItem] = Field(..., min_length=1, max_length=1000)


class PeopleBatchCreateResponse(_PublicBaseModel):
    request_id: str
    count: int
    people: list[PersonSummary]


class BulkEnrollPhotoItem(_PublicBaseModel):
    filename: str = Field(..., min_length=1)
    image_base64: str = Field(..., min_length=1)


class BulkEnrollIdentityItem(_PublicBaseModel):
    display_name: str = Field(..., min_length=1)
    photos: list[BulkEnrollPhotoItem] = Field(..., min_length=1, max_length=100)
    metadata: dict[str, Any] | None = None
    source_dataset: str | None = None


class BulkEnrollRequest(_PublicBaseModel):
    identities: list[BulkEnrollIdentityItem] = Field(..., min_length=1, max_length=1000)


class BulkEnrollResponse(_PublicBaseModel):
    request_id: str
    process_id: str
    discovered_identities: int
    discovered_photos: int
    enrolled_identities: int
    enrolled_photos: int
    no_face: int
    decode_error: int
    failed: int
    errors: list[str] = Field(default_factory=list)


class PersonListResponse(_PublicBaseModel):
    request_id: str
    count: int
    people: list[PersonSummary]


class PersonDetailResponse(_PublicBaseModel):
    request_id: str
    person_id: str
    display_name: str
    is_active: bool
    metadata: dict[str, Any]
    face_count: int = 0
    faces: list[IdentitySummary] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None


class FaceSampleResponse(_PublicBaseModel):
    sample_id: str
    face_id: str
    state: str
    image_url: str | None = None
    created_at: str | None = None
    activated_at: str | None = None


class FaceSamplesResponse(_PublicBaseModel):
    request_id: str
    face_id: str
    count: int
    samples: list[FaceSampleResponse]


class RecognitionResultSummary(_PublicBaseModel):
    result_id: str
    face_id: str
    status: str
    confidence: float


class ProcessResponse(_PublicBaseModel):
    request_id: str
    process_id: str
    process_type: str
    status: str
    face_count: int | None = None
    error_code: str | None = None
    details: dict[str, Any] | None = None
    created_at: str | None = None
    completed_at: str | None = None
    failed_at: str | None = None
    results: list[RecognitionResultSummary] | None = None


class EmptyResponse(_PublicBaseModel):
    request_id: str


class VideoRecognizeRequest(_PublicBaseModel):
    sampling_mode: str = "every_frame"
    every_n_frames: int | None = None
    frames_per_second: Decimal | None = None


class VideoRecognizeResponse(_PublicBaseModel):
    request_id: str
    process_id: str
    video_id: str
    job_id: str
    upload_session_id: str | None = None
    status: str
    status_url: str
    result_url: str


class VideoResponse(_PublicBaseModel):
    request_id: str
    video_id: str
    upload_session_id: str
    state: str
    content_sha256: str | None = None
    size_bytes: int | None = None
    container_format: str | None = None
    video_codec: str | None = None
    display_width: int | None = None
    display_height: int | None = None
    rotation_degrees: int = 0
    duration_ns: int | None = None
    total_frames: int | None = None
    failure_code: str | None = None


class VideoJobResponse(_PublicBaseModel):
    request_id: str
    process_id: str
    video_id: str
    job_id: str
    state: str
    stage: str
    progress_percent: int
    sampling_mode: str
    every_n_frames: int | None = None
    frames_per_second: Decimal | None = None
    processed_frames: int
    sampled_frames: int
    detected_observations: int
    person_count: int
    cancellation_requested: bool = False
    error_code: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    status_url: str
    result_url: str


class VideoRetryResponse(VideoRecognizeResponse):
    pass


class VideoJobResultResponse(_PublicBaseModel):
    request_id: str
    job_id: str
    state: str
    result_available: bool
    manifest_bucket: str | None = None
    manifest_key: str | None = None
    manifest_sha256: str | None = None


class VideoPersonSummary(_PublicBaseModel):
    track_id: str
    face_id: str
    status: str
    name: str | None = None
    current_status: str | None = None
    current_name: str | None = None
    first_frame_index: int
    last_frame_index: int
    first_pts_ns: int
    last_pts_ns: int
    total_duration_ns: int
    detection_count: int
    appearance_count: int
    match_confidence: float


class VideoPeopleResponse(_PublicBaseModel):
    request_id: str
    job_id: str
    person_count: int
    people: list[VideoPersonSummary]


class VideoAppearanceEntry(_PublicBaseModel):
    track_id: str
    face_id: str
    start_frame_index: int
    end_frame_index: int
    start_pts_ns: int
    end_pts_ns: int
    detection_count: int


class VideoAppearancesResponse(_PublicBaseModel):
    request_id: str
    job_id: str
    appearance_count: int
    appearances: list[VideoAppearanceEntry]


class VideoTimelineRecord(_PublicBaseModel):
    track_id: str
    face_id: str
    start_frame_index: int
    end_frame_index: int
    start_pts_ns: int
    end_pts_ns: int


class VideoTimelineResponse(_PublicBaseModel):
    request_id: str
    job_id: str
    record_count: int
    records: list[VideoTimelineRecord]


class OverlayBoundingBox(_PublicBaseModel):
    x: int
    y: int
    width: int
    height: int


class OverlayDetection(_PublicBaseModel):
    track_id: str
    face_id: str
    status: str
    name: str | None
    bbox: OverlayBoundingBox
    confidence: float
    provenance: str


class OverlayFrame(_PublicBaseModel):
    frame_index: int
    pts_ns: int
    detections: list[OverlayDetection]


class VideoTimelineFramesResponse(_PublicBaseModel):
    request_id: str
    job_id: str
    start_pts_ns: int
    end_pts_ns: int
    record_count: int
    frames: list[OverlayFrame]
