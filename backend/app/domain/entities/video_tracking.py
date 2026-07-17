"""Tracking domain entities."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from app.domain.value_objects import BoundingBox


@dataclass(frozen=True)
class TrackDetection:
    detection_id: str
    frame_index: int
    pts_ns: int
    bbox: BoundingBox
    landmarks: tuple[float, ...]
    detector_score: float
    quality_score: float
    embedding: tuple[float, ...]
    raw_track_key: str = ""


@dataclass
class RawTracklet:
    tracklet_id: uuid.UUID
    job_id: uuid.UUID
    ordinal: int
    state: str = "confirmed"
    detections: list[TrackDetection] = field(default_factory=list)
    mean_quality: float | None = None
    max_quality: float | None = None

    def __post_init__(self) -> None:
        if self.state not in {"confirmed", "lost", "removed"}:
            raise ValueError(f"Invalid tracklet state: {self.state}")


@dataclass(frozen=True)
class TrackletTemplate:
    tracklet_id: uuid.UUID
    sample_indices: list[int]
    qualities: list[float]


@dataclass(frozen=True)
class AppearanceInterval:
    start_frame_index: int
    end_frame_index: int
    start_pts_ns: int
    end_pts_ns: int
    detection_count: int = 0


@dataclass
class CanonicalTrack:
    track_id: uuid.UUID
    tracklets: list[RawTracklet] = field(default_factory=list)
    cannot_link_track_ids: set[uuid.UUID] = field(default_factory=set)
    representative_embedding: tuple[float, ...] = field(default_factory=tuple)
    appearance_intervals: list[AppearanceInterval] = field(default_factory=list)
