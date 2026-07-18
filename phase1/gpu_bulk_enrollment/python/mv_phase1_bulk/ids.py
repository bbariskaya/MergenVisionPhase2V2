"""Deterministic/idempotent identifier generation for Phase 1 bulk enrollment.

Phase 2 compatibility:
- All IDs are string UUIDs so they fit ``PersonOrm.person_id``,
  ``FaceIdentityOrm.face_id`` and ``FaceSampleOrm.sample_id``.
- Object keys follow the Phase 2 contract:
  ``faces/{face_id}/{sample_id}/original.jpg``.
- Determinism guarantees idempotent re-runs.

Privacy:
- The raw ``external_subject_key`` never leaves this module.
- Logs, object keys, Qdrant payloads and reports only see the resulting UUIDs.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from typing import NewType

_NS_PERSON = uuid.UUID("018f5e1a-7b00-7e0a-8f0c-7c3b8e1a5f00")
_NS_FACE = uuid.UUID("018f5e1a-7b00-7e0a-8f0c-7c3b8e1a5f01")
_NS_SAMPLE = uuid.UUID("018f5e1a-7b00-7e0a-8f0c-7c3b8e1a5f02")

PersonId = NewType("PersonId", str)
FaceId = NewType("FaceId", str)
SampleId = NewType("SampleId", str)
RunId = NewType("RunId", str)
ExternalSubjectKey = NewType("ExternalSubjectKey", str)


class IdNamespaceError(RuntimeError):
    """Raised when the HMAC key fingerprint does not match a previous run."""


def _require_hmac_key() -> bytes:
    key = os.environ.get("MV_PHASE1_BULK_ID_HMAC_KEY")
    if not key:
        raise IdNamespaceError(
            "MV_PHASE1_BULK_ID_HMAC_KEY environment variable is required for deterministic ID generation"
        )
    return key.encode("utf-8")


def hmac_key_fingerprint() -> str:
    """Public fingerprint of the current HMAC key for run-journal comparison."""
    key = _require_hmac_key()
    return hashlib.sha256(key).hexdigest()[:32]


def _identity_hmac(source_namespace: str, external_subject_key: str) -> str:
    key = _require_hmac_key()
    name = f"{source_namespace}:{external_subject_key.strip().lower()}"
    return hmac.new(key, name.encode("utf-8"), hashlib.sha256).hexdigest()


def _uuid5(namespace: uuid.UUID, name: str) -> uuid.UUID:
    return uuid.uuid5(namespace, name)


def make_person_id(
    source_namespace: str,
    external_subject_key: ExternalSubjectKey | str,
) -> PersonId:
    """Deterministic person id for an external subject."""
    h = _identity_hmac(source_namespace, external_subject_key)
    return PersonId(str(_uuid5(_NS_PERSON, h)))


def make_face_id(
    source_namespace: str,
    external_subject_key: ExternalSubjectKey | str,
) -> FaceId:
    """Deterministic face identity id.

    One person has exactly one face identity in Phase 1 bulk enrollment.
    The face id is independent of model/preprocess version.
    """
    h = _identity_hmac(source_namespace, external_subject_key)
    return FaceId(str(_uuid5(_NS_FACE, h)))


def make_sample_id(
    face_id: FaceId | str,
    image_sha256: str,
    model_version: str,
    preprocess_version: str,
) -> SampleId:
    """Deterministic sample id.

    The same image bytes for the same face always produce the same sample id,
    making the pipeline idempotent across retries.
    """
    name = f"{face_id}:{image_sha256}:{model_version}:{preprocess_version}"
    return SampleId(str(_uuid5(_NS_SAMPLE, name)))


def make_object_key(face_id: FaceId | str, sample_id: SampleId | str) -> str:
    """Phase 2 object key for the original input photo (JPEG)."""
    return f"faces/{face_id}/{sample_id}/original.jpg"


def normalize_uuid(value: str) -> str:
    """Validate and canonicalize a UUID string."""
    return str(uuid.UUID(value))
