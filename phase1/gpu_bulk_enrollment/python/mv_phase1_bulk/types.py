"""Public data types for Phase 1 GPU bulk extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FaceExtraction:
    source_index: int
    detection_ordinal: int
    bbox_original: np.ndarray  # [x1, y1, x2, y2]
    landmarks_original: np.ndarray  # [5, 2]
    detector_score: float
    quality_primitives: dict[str, float] = field(default_factory=dict)
    embedding: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    embedding_norm: float = 1.0
    crop_bytes: bytes = b""
    model_version: str = ""
    preprocess_version: str = ""

    def to_compact_dict(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "detection_ordinal": self.detection_ordinal,
            "bbox_original": self.bbox_original.tolist(),
            "landmarks_original": self.landmarks_original.tolist(),
            "detector_score": self.detector_score,
            "quality_primitives": self.quality_primitives,
            "embedding": self.embedding.tolist(),
            "embedding_norm": self.embedding_norm,
            "model_version": self.model_version,
            "preprocess_version": self.preprocess_version,
        }


@dataclass
class ImageExtractionResult:
    source_index: int
    source_key: str
    original_width: int
    original_height: int
    status: str  # accepted, no_face, quarantine, decode_failed, inference-error, low_quality
    rejection_reason: str | None = None
    faces: list[FaceExtraction] = field(default_factory=list)

    def to_compact_dict(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "source_key": self.source_key,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "faces": [f.to_compact_dict() for f in self.faces],
        }


@dataclass
class FaceRecord:
    """Phase 2 compatible face_identity row."""

    face_id: str
    status: str = "known"  # known | anonymous
    is_active: bool = True
    display_name: str = ""
    identity_metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1


@dataclass
class SampleRecord:
    """Phase 2 compatible face_sample row."""

    sample_id: str
    face_id: str
    state: str = "pending"  # pending | active | failed | inactive
    bucket: str | None = None
    object_key: str | None = None
    failure_code: str | None = None
    is_active: bool = False
    activated_at: str | None = None
    deactivated_at: str | None = None


@dataclass
class EnrollmentBundle:
    """A single subject ready for cross-store persistence."""

    external_subject_key: str
    display_name: str
    face_id: str
    model_version: str
    samples: list[SampleRecord] = field(default_factory=list)


@dataclass
class EnrollmentOutcome:
    """Result of persisting one bundle."""

    external_subject_key: str
    face_id: str
    persisted_sample_ids: list[str] = field(default_factory=list)
    failed_sample_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_compact_dict(self) -> dict[str, Any]:
        return {
            "external_subject_key": self.external_subject_key,
            "face_id": self.face_id,
            "persisted_sample_ids": self.persisted_sample_ids,
            "failed_sample_ids": self.failed_sample_ids,
        }
