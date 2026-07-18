"""Build enrollment bundles from a validated manifest.

A bundle ties together one deterministic known ``FaceIdentity`` and the set of
``FaceSample`` rows derived from that subject's images.  It intentionally keeps
the raw ``external_subject_key`` inside this module; downstream stores only see
UUIDs and object keys.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mv_phase1_bulk.ids import make_face_id, make_object_key, make_sample_id
from mv_phase1_bulk.manifest import EnrollmentManifest
from mv_phase1_bulk.types import FaceRecord, SampleRecord


@dataclass
class EnrolledSample:
    """Runtime enrichment of a ``SampleRecord`` with extraction outputs."""

    sample_record: SampleRecord
    image_sha256: str
    image_bytes: bytes = b""
    crop_bytes: bytes = b""
    embedding: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    def set_extraction(self, crop_bytes: bytes, embedding: np.ndarray) -> None:
        self.crop_bytes = crop_bytes
        self.embedding = embedding


@dataclass
class SubjectBundle:
    """Everything needed to persist one subject across PG/MinIO/Qdrant."""

    face: FaceRecord
    samples: list[EnrolledSample] = field(default_factory=list)


def _display_name(subject_key: str) -> str:
    """Derive a display name while never forwarding raw path fragments."""
    return subject_key.strip() or "unknown"


def _metadata(source_namespace: str) -> dict[str, Any]:
    """Fingerprint-only metadata; never include the raw external key."""
    return {"source_namespace": source_namespace}


def build_subject_bundles(
    manifest: EnrollmentManifest,
    *,
    source_namespace: str,
    model_version: str,
    preprocess_version: str,
) -> list[SubjectBundle]:
    """Create one bundle per manifest subject with deterministic IDs.

    ``crop_bytes`` and ``embedding`` are left empty; the extraction stage
    fills them via ``EnrolledSample.set_extraction``.

    TODO: stream manifest records and read image bytes lazily to avoid loading
    the entire dataset into RAM.
    """
    bundles: list[SubjectBundle] = []
    for record in manifest:
        face_id = make_face_id(source_namespace, record.subject_key)
        display_name = _display_name(record.subject_key)

        face = FaceRecord(
            face_id=face_id,
            status="known",
            is_active=True,
            display_name=display_name,
            identity_metadata=_metadata(source_namespace),
        )

        enrolled_samples: list[EnrolledSample] = []
        for image_path in record.image_paths:
            data = Path(image_path).read_bytes()
            image_sha256 = hashlib.sha256(data).hexdigest()
            sample_id = make_sample_id(face_id, image_sha256, model_version, preprocess_version)
            object_key = make_object_key(face_id, sample_id)
            sample = SampleRecord(
                sample_id=sample_id,
                face_id=face_id,
                state="pending",
                object_key=object_key,
            )
            enrolled_samples.append(
                EnrolledSample(
                    sample_record=sample,
                    image_sha256=image_sha256,
                    image_bytes=data,
                )
            )

        bundles.append(SubjectBundle(face=face, samples=enrolled_samples))

    return bundles
