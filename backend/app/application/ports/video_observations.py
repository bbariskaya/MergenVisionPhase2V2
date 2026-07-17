"""Lightweight observation DTOs matching video_observation_v1.proto.

These are used by the Python tracker so that observations can come either from
protobuf/zstd artifacts emitted by the native worker or from synthetic test
fixtures without leaking proto codecs into the domain.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.value_objects import BoundingBox


@dataclass(frozen=True)
class FaceObservation:
    detection_id: str
    ordinal: int
    bbox: BoundingBox
    landmarks: tuple[float, ...]
    detector_score: float
    quality_score: float
    tracking_eligible: bool
    recognition_eligible: bool
    rejection_code: str
    embedding: tuple[float, ...]
    model_version: str
    preprocess_version: str
    raw_track_key: str = ""


@dataclass(frozen=True)
class VideoObservationFrame:
    job_id: str
    video_id: str
    stream_index: int
    frame_index: int
    source_pts: int
    pts_ns: int
    display_width: int
    display_height: int
    detections: tuple[FaceObservation, ...]
