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


@dataclass
class ImageExtractionResult:
    source_index: int
    source_key: str
    original_width: int
    original_height: int
    status: str  # accepted, no_face, multi_face, decode_failed, inference-error, ...
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
            "faces": [
                {
                    "source_index": f.source_index,
                    "detection_ordinal": f.detection_ordinal,
                    "bbox_original": f.bbox_original.tolist(),
                    "landmarks_original": f.landmarks_original.tolist(),
                    "detector_score": f.detector_score,
                    "quality_primitives": f.quality_primitives,
                    "embedding": f.embedding.tolist(),
                    "embedding_norm": f.embedding_norm,
                    "model_version": f.model_version,
                    "preprocess_version": f.preprocess_version,
                }
                for f in self.faces
            ],
        }


@dataclass
class PersonRecord:
    """Phase 2 compatible person row."""

    person_id: str
    display_name: str
    status: str = "active"  # active | inactive
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FaceRecord:
    """Phase 2 compatible face_identity row."""

    face_id: str
    person_id: str
    model_version: str
    status: str = "active"  # active | inactive
    is_canonical: bool = True
    # ``display_name`` and ``metadata`` are mirrored from person for Phase 2 search.
    display_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleRecord:
    """Phase 2 compatible face_sample row."""

    sample_id: str
    face_id: str
    person_id: str
    status: str = "pending"  # pending | active | failed | inactive
    bucket: str | None = None
    object_key: str | None = None
    sha256: str = ""
    model_version: str = ""
    preprocess_version: str = ""
    rejection_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrollmentBundle:
    """A single subject ready for cross-store persistence."""

    external_subject_key: str
    display_name: str
    person_id: str
    face_id: str
    model_version: str
    samples: list[SampleRecord] = field(default_factory=list)


@dataclass
class EnrollmentOutcome:
    """Result of persisting one bundle."""

    external_subject_key: str
    person_id: str
    face_id: str
    persisted_sample_ids: list[str] = field(default_factory=list)
    failed_sample_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

        return {
            "source_index": self.source_index,
            "source_key": self.source_key,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "faces": [
                {
                    "source_index": f.source_index,
                    "detection_ordinal": f.detection_ordinal,
                    "bbox_original": f.bbox_original.tolist(),
                    "landmarks_original": f.landmarks_original.tolist(),
                    "detector_score": f.detector_score,
                    "quality_primitives": f.quality_primitives,
                    "embedding": f.embedding.tolist(),
                    "embedding_norm": f.embedding_norm,
                    "model_version": f.model_version,
                    "preprocess_version": f.preprocess_version,
                }
                for f in self.faces
            ],
        }
