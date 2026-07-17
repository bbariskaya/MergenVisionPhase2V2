"""Request and response contracts for the MergenVision API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BoundingBoxSchema(BaseModel):
    x: int
    y: int
    width: int
    height: int


class FaceResponse(BaseModel):
    face_id: str
    status: str = Field(..., pattern=r"^(known|anonymous|new_anonymous)$")
    name: str | None
    metadata: dict[str, Any] | None
    bounding_box: BoundingBoxSchema
    confidence: float = Field(..., ge=0.0, le=1.0)


class RecognizeResponse(BaseModel):
    process_id: str
    status: str
    face_count: int
    faces: list[FaceResponse]


class EnrollRequest(BaseModel):
    face_id: str
    name: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = None


class EnrollResponse(BaseModel):
    face_id: str
    status: str
    name: str
    metadata: dict[str, Any]


class IdentityDetailResponse(BaseModel):
    face_id: str
    status: str
    name: str | None
    metadata: dict[str, Any]
    created_at: str | None = None
    updated_at: str | None = None


class HistoryEntry(BaseModel):
    process_id: str
    timestamp: str
    process_type: str | None = None
    status: str | None = None


class FaceHistoryResponse(BaseModel):
    face_id: str
    history: list[HistoryEntry]


class ProcessResponse(BaseModel):
    process_id: str
    process_type: str
    status: str
    face_count: int | None
    error_code: str | None = None
    details: dict[str, Any] | None = None
    created_at: str | None = None
    completed_at: str | None = None
