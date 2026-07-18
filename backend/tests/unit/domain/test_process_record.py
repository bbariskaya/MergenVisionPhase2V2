"""Unit tests for ProcessRecord state transitions."""

import pytest

from app.domain.entities.process_record import ProcessRecord
from app.domain.errors import InvalidTransitionError
from app.domain.value_objects import ProcessId
from app.infrastructure.uuid7 import generate_uuid7


def _process(process_type: str = "image_recognize") -> ProcessRecord:
    return ProcessRecord(process_id=ProcessId(generate_uuid7()), process_type=process_type)


def test_new_process_is_processing() -> None:
    process = _process()
    assert process.status == "processing"


def test_complete_sets_status_and_count() -> None:
    process = _process()
    process.complete(face_count=1, details={"face_ids": []})
    assert process.status == "completed"
    assert process.face_count == 1
    assert process.completed_at is not None


def test_fail_sets_error() -> None:
    process = _process()
    process.fail("minio_timeout", details={"key": "faces/face/sample.webp"})
    assert process.status == "failed"
    assert process.error_code == "minio_timeout"
    assert process.failed_at is not None
    assert process.completed_at is None


def test_cancel_sets_cancelled_at() -> None:
    process = _process()
    process.cancel(details={"reason": "user_request"})
    assert process.status == "cancelled"
    assert process.cancelled_at is not None
    assert process.completed_at is None
    assert process.failed_at is None
    assert process.error_code is None


def test_cannot_complete_completed_process() -> None:
    process = _process()
    process.complete(1)
    with pytest.raises(InvalidTransitionError):
        process.complete(2)
