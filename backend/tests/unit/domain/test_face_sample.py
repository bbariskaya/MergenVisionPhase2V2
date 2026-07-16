"""Unit tests for FaceSample state transitions."""

import uuid

import pytest

from app.domain.entities.face_sample import FaceSample
from app.domain.errors import InvalidTransitionError
from app.domain.value_objects import FaceId, SampleId


def _ids() -> tuple[FaceId, SampleId]:
    return FaceId(uuid.uuid4()), SampleId(uuid.uuid4())


def test_new_sample_is_pending() -> None:
    face_id, sample_id = _ids()
    sample = FaceSample(sample_id=sample_id, face_id=face_id)
    assert sample.state == "pending"
    assert sample.is_active is True


def test_mark_active_transitions_from_pending() -> None:
    face_id, sample_id = _ids()
    sample = FaceSample(sample_id=sample_id, face_id=face_id)
    sample.mark_active("bucket", "faces/face/sample.webp")
    assert sample.state == "active"
    assert sample.bucket == "bucket"
    assert sample.object_key == "faces/face/sample.webp"
    assert sample.activated_at is not None


def test_mark_failed_from_pending() -> None:
    face_id, sample_id = _ids()
    sample = FaceSample(sample_id=sample_id, face_id=face_id)
    sample.mark_failed("minio_timeout")
    assert sample.state == "failed"
    assert sample.is_active is False
    assert sample.failure_code == "minio_timeout"


def test_mark_inactive_from_active() -> None:
    face_id, sample_id = _ids()
    sample = FaceSample(sample_id=sample_id, face_id=face_id)
    sample.mark_active("bucket", "key")
    sample.mark_inactive()
    assert sample.state == "inactive"
    assert sample.is_active is False
    assert sample.deactivated_at is not None


def test_cannot_activate_failed_sample() -> None:
    face_id, sample_id = _ids()
    sample = FaceSample(sample_id=sample_id, face_id=face_id)
    sample.mark_failed("minio_timeout")
    with pytest.raises(InvalidTransitionError):
        sample.mark_active("bucket", "key")
