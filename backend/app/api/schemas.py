"""Request and response contracts for the MergenVision API.

All public response fields use camelCase aliases. Internal Python code keeps
snake_case names. Use `populate_by_name=True` so model constructors accept both
forms during tests and service code.
"""

from __future__ import annotations

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


class EnrollResponse(_PublicBaseModel):
    request_id: str
    process_id: str
    face_id: str
    status: str
    name: str
    metadata: dict[str, Any]


class IdentityDetailResponse(_PublicBaseModel):
    request_id: str
    face_id: str
    status: str
    name: str | None = None
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
