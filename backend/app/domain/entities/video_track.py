"""Video result domain entities."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from app.domain.value_objects import FaceId, JobId, ResultId, SampleId


@dataclass(frozen=True)
class VideoAppearanceInterval:
    appearance_id: uuid.UUID
    job_id: JobId
    track_id: uuid.UUID
    interval_index: int
    start_frame_index: int
    end_frame_index: int
    start_pts_ns: int
    end_pts_ns: int
    detection_count: int


@dataclass(frozen=True)
class VideoTracklet:
    tracklet_id: uuid.UUID
    job_id: JobId
    track_id: uuid.UUID
    tracklet_ordinal: int
    first_frame_index: int
    last_frame_index: int
    first_pts_ns: int
    last_pts_ns: int
    observation_count: int
    valid_embedding_count: int
    state: str
    mean_quality: float | None
    max_quality: float | None


@dataclass(frozen=True)
class VideoTrackSample:
    track_id: uuid.UUID
    sample_id: SampleId
    sample_rank: int
    quality_score: float
    purpose: str


@dataclass
class VideoTrack:
    track_id: uuid.UUID
    job_id: JobId
    track_ordinal: int
    face_id: FaceId
    recognition_result_id: ResultId
    status_at_processing: str
    name_at_processing: str | None
    metadata_at_processing: dict[str, Any]
    identity_version_at_processing: int
    match_confidence: float
    top1_score: float | None
    top2_score: float | None
    margin_score: float | None
    threshold_used: float | None
    first_frame_index: int
    last_frame_index: int
    first_pts_ns: int
    last_pts_ns: int
    total_duration_ns: int
    detection_count: int
    tracklet_count: int
    best_sample_id: SampleId | None = None
