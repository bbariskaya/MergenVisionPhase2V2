"""Tests for deterministic Phase 1 bulk enrollment identifiers."""

from __future__ import annotations

from typing import Any

import pytest
from mv_phase1_bulk.ids import (
    hmac_key_fingerprint,
    make_face_id,
    make_object_key,
    make_person_id,
    make_sample_id,
    normalize_uuid,
)


@pytest.fixture(autouse=True)
def _hmac_key(monkeypatch: Any) -> None:
    monkeypatch.setenv("MV_PHASE1_BULK_ID_HMAC_KEY", "unit-test-key")


def test_person_id_is_deterministic() -> None:
    a = make_person_id("ns", "subject_A")
    b = make_person_id("ns", "subject_A")
    assert a == b
    assert normalize_uuid(a) == a


def test_face_id_differs_from_person_id() -> None:
    person_id = make_person_id("ns", "subject_A")
    face_id = make_face_id("ns", "subject_A")
    assert face_id != person_id
    # Both are valid UUIDs.
    assert normalize_uuid(person_id) == person_id
    assert normalize_uuid(face_id) == face_id


def test_same_subject_same_face() -> None:
    face_a = make_face_id("ns", "subject_A")
    face_b = make_face_id("ns", "subject_A")
    assert face_a == face_b


def test_namespace_isolates_subjects() -> None:
    face_ns1 = make_face_id("ns1", "subject_A")
    face_ns2 = make_face_id("ns2", "subject_A")
    assert face_ns1 != face_ns2


def test_subject_key_is_case_normalized() -> None:
    lower = make_face_id("ns", "subject_a")
    upper = make_face_id("ns", "SUBJECT_A")
    mixed = make_face_id("ns", " Subject_A ")
    assert lower == upper == mixed


def test_sample_id_includes_image_and_versions() -> None:
    face_id = make_face_id("ns", "subject_A")
    sample_a = make_sample_id(face_id, "sha_a", "model_v1", "pre_v1")
    sample_b = make_sample_id(face_id, "sha_a", "model_v1", "pre_v1")
    assert sample_a == sample_b

    different_sha = make_sample_id(face_id, "sha_b", "model_v1", "pre_v1")
    different_model = make_sample_id(face_id, "sha_a", "model_v2", "pre_v1")
    different_pre = make_sample_id(face_id, "sha_a", "model_v1", "pre_v2")
    assert different_sha != sample_a
    assert different_model != sample_a
    assert different_pre != sample_a


def test_object_key_follows_phase2_contract() -> None:
    face_id = make_face_id("ns", "subject_A")
    sample_id = make_sample_id(face_id, "sha", "model_v1", "pre_v1")
    key = make_object_key(face_id, sample_id)
    assert key == f"faces/{face_id}/{sample_id}/original.jpg"


def test_hmac_fingerprint_is_stable() -> None:
    assert hmac_key_fingerprint() == hmac_key_fingerprint()
    assert len(hmac_key_fingerprint()) == 32
