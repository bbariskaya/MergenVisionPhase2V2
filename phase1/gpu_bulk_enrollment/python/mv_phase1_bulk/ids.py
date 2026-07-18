"""Deterministic/idempotent identifier generation for Phase 1 bulk enrollment.

Phase 2 compatibility:
- All IDs are string UUIDs (``str(uuid.UUID)``) so they fit ``PersonOrm.person_id``,
  ``FaceIdentityOrm.face_id`` and ``FaceSampleOrm.sample_id``.
- Object keys follow the Phase 2 contract:
  ``faces/{face_id}/{sample_id}/aligned.webp``.
- Determinism guarantees idempotent re-runs: the same manifest entry always yields
  the same person/face/sample IDs, so PostgreSQL/Qdrant/MinIO upserts are safe.
"""

from __future__ import annotations

import uuid
from typing import NewType

# Phase 1 bulk enrollment namespaces (version-5 UUIDs).
_NS_PERSON = uuid.UUID("018f5e1a-7b00-7e0a-8f0c-7c3b8e1a5f00")
_NS_FACE = uuid.UUID("018f5e1a-7b00-7e0a-8f0c-7c3b8e1a5f01")
_NS_SAMPLE = uuid.UUID("018f5e1a-7b00-7e0a-8f0c-7c3b8e1a5f02")

PersonId = NewType("PersonId", str)
FaceId = NewType("FaceId", str)
SampleId = NewType("SampleId", str)
RunId = NewType("RunId", str)
ExternalSubjectKey = NewType("ExternalSubjectKey", str)


def new_run_id() -> RunId:
    """Return a fresh time-ordered UUIDv7 run id."""
    # Prefer uuid_extensions when available to match Phase 2 exactly.
    try:
        from uuid_extensions import uuid7  # type: ignore[import-untyped]

        return RunId(str(uuid.UUID(str(uuid7()))))
    except Exception:
        # Fallback: UUIDv4 is acceptable only for run ids (not entity ids).
        return RunId(str(uuid.uuid4()))


def _uuid5(namespace: uuid.UUID, name: str) -> uuid.UUID:
    return uuid.uuid5(namespace, name)


def make_person_id(
    source_namespace: str,
    external_subject_key: ExternalSubjectKey | str,
) -> PersonId:
    """Deterministic person id for an external subject."""
    name = f"{source_namespace}:{external_subject_key}"
    return PersonId(str(_uuid5(_NS_PERSON, name)))


def make_face_id(
    person_id: PersonId | str,
    model_version: str,
) -> FaceId:
    """Deterministic face identity id.

    One person has exactly one face per model version in Phase 1 bulk enrollment.
    """
    name = f"{person_id}:{model_version}"
    return FaceId(str(_uuid5(_NS_FACE, name)))


def make_sample_id(
    face_id: FaceId | str,
    image_sha256: str,
) -> SampleId:
    """Deterministic sample id.

    The same image bytes for the same face always produce the same sample id,
    making the pipeline idempotent across retries.
    """
    name = f"{face_id}:{image_sha256}"
    return SampleId(str(_uuid5(_NS_SAMPLE, name)))


def make_object_key(face_id: FaceId | str, sample_id: SampleId | str) -> str:
    """Phase 2 canonical object key for an aligned face crop."""
    return f"faces/{face_id}/{sample_id}/aligned.webp"


def normalize_uuid(value: str) -> str:
    """Validate and canonicalize a UUID string."""
    return str(uuid.UUID(value))
